"""Read the original UCI archive; audit, aggregate and build causal features."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

POLLUTANTS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
WEATHER = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
WIND = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
        'S','SSW','SW','WSW','W','WNW','NW','NNW']
TEST_START = pd.Timestamp('2016-03-01')

class DataValidationError(ValueError):
    """Input data violate the declared data contract."""

class AirQualityDataset:
    """Encapsulate original data, validation rules and daily aggregation."""
    def __init__(self, archive, min_hours=18):
        self.archive = Path(archive)
        if not isinstance(min_hours, int) or not 1 <= min_hours <= 24:
            raise ValueError('min_hours must be an integer between 1 and 24')
        self.min_hours = min_hours
        self.audit = {}

    def load(self):
        if not self.archive.is_file():
            raise FileNotFoundError(f'Data archive not found: {self.archive}')
        try:
            with zipfile.ZipFile(self.archive) as outer:
                nested = [n for n in outer.namelist() if n.startswith('PRSA2017') and n.endswith('.zip')]
                if len(nested) != 1:
                    raise DataValidationError('Expected one official PRSA2017 inner archive')
                with zipfile.ZipFile(BytesIO(outer.read(nested[0]))) as inner:
                    names = sorted(n for n in inner.namelist() if n.endswith('.csv') and '/.' not in n)
                    df = pd.concat([pd.read_csv(inner.open(n)) for n in names], ignore_index=True)
        except (zipfile.BadZipFile, pd.errors.ParserError) as exc:
            raise DataValidationError(f'Cannot parse the UCI archive: {exc}') from exc
        required = {'station','year','month','day','hour','wd', *POLLUTANTS, *WEATHER}
        if missing := required.difference(df.columns):
            raise DataValidationError(f'Missing required columns: {sorted(missing)}')
        df['timestamp'] = pd.to_datetime(df[['year','month','day','hour']], errors='raise')
        duplicates = int(df.duplicated(['station','timestamp']).sum())
        if duplicates:
            raise DataValidationError(f'{duplicates} duplicated station-hour keys; inspect before proceeding')
        invalid = {}
        # Negative temperature and dew point are valid; negative concentrations are not.
        for col in POLLUTANTS + ['RAIN','WSPM']:
            mask = df[col].lt(0)
            invalid[col] = int(mask.sum())
            df.loc[mask, col] = np.nan
        bad_wind = df['wd'].notna() & ~df['wd'].isin(WIND)
        invalid['wd'] = int(bad_wind.sum())
        df.loc[bad_wind, 'wd'] = np.nan
        expected = pd.date_range(df.timestamp.min(), df.timestamp.max(), freq='h')
        self.audit = {
            'raw_rows': len(df), 'raw_columns': len(df.columns)-1,
            'stations': sorted(df.station.unique()), 'station_count': df.station.nunique(),
            'start': str(df.timestamp.min()), 'end': str(df.timestamp.max()),
            'duplicates': duplicates, 'invalid_to_nan': invalid,
            'missing_hour_keys': len(expected)*df.station.nunique()-len(df),
            'missing_counts': df[POLLUTANTS+WEATHER+['wd']].isna().sum().to_dict(),
            'missing_pct': (100*df[POLLUTANTS+WEATHER+['wd']].isna().mean()).round(4).to_dict(),
            'sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            'min_hours': self.min_hours,
        }
        return df.sort_values(['station','timestamp']).reset_index(drop=True)

    def aggregate(self, hourly):
        df = hourly.copy()
        df['date'] = df.timestamp.dt.normalize()
        angles = df.wd.map({w: i*np.pi/8 for i,w in enumerate(WIND)})
        df['wind_sin'], df['wind_cos'] = np.sin(angles), np.cos(angles)
        fields = POLLUTANTS + WEATHER + ['wind_sin','wind_cos']
        g = df.groupby(['station','date'], observed=True)
        daily = g[fields].mean().where(g[fields].count() >= self.min_hours)
        daily['RAIN'] = g.RAIN.sum(min_count=self.min_hours)
        daily['pm_hours'] = g['PM2.5'].count()
        daily['pm_max'] = g['PM2.5'].max().where(daily.pm_hours >= self.min_hours)
        daily['pm_std'] = g['PM2.5'].std().where(daily.pm_hours >= self.min_hours)
        late = df[df.hour >= 18].groupby(['station','date'])['PM2.5']
        daily['pm_evening'] = late.mean().where(late.count() >= 4)
        # Complete station-date grid prevents shift(-1) from jumping over absent dates.
        grid = pd.MultiIndex.from_product([sorted(df.station.unique()),
                    pd.date_range(df.date.min(), df.date.max(), freq='D')], names=['station','date'])
        daily = daily.reindex(grid).reset_index()
        self.audit.update(daily_rows=len(daily), invalid_pm_days=int(daily['PM2.5'].isna().sum()))
        return daily

def build_features(daily):
    """At the end of day t predict mean PM2.5 on calendar day t+1."""
    d = daily.sort_values(['station','date']).copy()
    g = d.groupby('station', sort=False)
    d['target'] = g['PM2.5'].shift(-1)
    d['target_hours'] = g.pm_hours.shift(-1)
    d['target_date'] = d.date + pd.Timedelta(days=1)
    for lag in [1,2,7]:
        d[f'pm_lag{lag}'] = g['PM2.5'].shift(lag)
    for window in [3,7]:
        d[f'pm_roll{window}'] = g['PM2.5'].transform(lambda s: s.rolling(window, min_periods=2).mean())
    d['pm_change'] = d['PM2.5'] - d.pm_lag1
    d['pressure_change'] = d.PRES - g.PRES.shift(1)
    city = d.groupby('date')['PM2.5'].agg(['mean','count'])
    d['city_pm'] = d.date.map(city['mean'].where(city['count'] >= 8))
    d['day_sin'] = np.sin(2*np.pi*d.target_date.dt.dayofyear/365.25)
    d['day_cos'] = np.cos(2*np.pi*d.target_date.dt.dayofyear/365.25)
    d['weekday'] = d.target_date.dt.dayofweek.astype(str)
    d['trend'] = (d.target_date - pd.Timestamp('2013-03-01')).dt.days
    calendar = ['station','weekday','day_sin','day_cos','trend']
    history = ['PM2.5','pm_hours','pm_max','pm_std','pm_evening',
               'pm_lag1','pm_lag2','pm_lag7','pm_roll3','pm_roll7','pm_change']
    weather = WEATHER + ['wind_sin','wind_cos','pressure_change']
    others = ['PM10','SO2','NO2','CO','O3','city_pm']
    groups = {'history_calendar': calendar+history, 'with_weather':calendar+history+weather,
              'full':calendar+history+weather+others}
    keep = d.target.notna() & d['PM2.5'].notna()
    d = d.loc[keep].sort_values(['target_date','station']).reset_index(drop=True)
    return d, groups

def calendar_folds(df):
    """Three expanding windows; all stations on a day stay in the same split."""
    folds, descriptions = [], []
    for start, end in [('2015-03-01','2015-07-01'),
                       ('2015-07-01','2015-11-01'),
                       ('2015-11-01','2016-03-01')]:
        begin, stop = pd.Timestamp(start), pd.Timestamp(end)
        train = np.flatnonzero(df.target_date < begin-pd.Timedelta(days=1))
        valid = np.flatnonzero((df.target_date >= begin) & (df.target_date < stop))
        if not len(train) or not len(valid):
            raise DataValidationError('A calendar fold has no samples')
        assert df.iloc[train].target_date.max() < df.iloc[valid].date.min()
        folds.append((train, valid))
        descriptions.append({'train_start':str(df.iloc[train].target_date.min().date()),
            'train_end':str(df.iloc[train].target_date.max().date()),
            'valid_start':str(df.iloc[valid].target_date.min().date()),
            'valid_end':str(df.iloc[valid].target_date.max().date()),
            'n_train':len(train),'n_valid':len(valid)})
    return folds, descriptions
