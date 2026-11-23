"""Meaningful checks for temporal correctness, data validation and saved results."""
from pathlib import Path
import tempfile
import unittest
import json
import numpy as np
import pandas as pd
from src.data import AirQualityDataset,DataValidationError,build_features,calendar_folds,TEST_START
from src.modeling import metrics,PersistenceRegressor

ROOT=Path(__file__).resolve().parents[1]

class ForecastIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily=pd.read_csv(ROOT/'data/processed/daily.csv',parse_dates=['date'])
        cls.frame,cls.groups=build_features(cls.daily)

    def test_target_is_exact_next_calendar_day(self):
        truth=self.daily.set_index(['station','date'])['PM2.5']
        expected=[truth.loc[(r.station,r.target_date)] for r in self.frame.itertuples()]
        np.testing.assert_allclose(self.frame.target,expected)

    def test_future_mutation_cannot_change_past_features(self):
        cutoff=pd.Timestamp('2015-06-15')
        altered=self.daily.copy()
        numeric=altered.select_dtypes('number').columns
        altered.loc[altered.date>cutoff,numeric]=9999
        changed,_=build_features(altered)
        before=self.frame[self.frame.date<=cutoff].set_index(['station','date'],drop=False)
        after=changed[changed.date<=cutoff].set_index(['station','date'],drop=False)
        common=before.index.intersection(after.index)
        pd.testing.assert_frame_equal(before.loc[common,self.groups['full']],after.loc[common,self.groups['full']])

    def test_all_station_dates_stay_together_and_labels_are_purged(self):
        dev=self.frame[self.frame.target_date<TEST_START-pd.Timedelta(days=1)].reset_index(drop=True)
        for tr,va in calendar_folds(dev)[0]:
            self.assertLess(dev.iloc[tr].target_date.max(),dev.iloc[va].date.min())
            self.assertFalse(set(dev.iloc[tr].target_date)&set(dev.iloc[va].target_date))
            dates=set(dev.iloc[va].target_date)
            self.assertEqual(len(va),int(dev.target_date.isin(dates).sum()))

    def test_no_target_columns_in_features(self):
        self.assertFalse(set(self.groups['full'])&{'target','target_hours','target_date'})
        self.assertTrue((self.frame.target_date-self.frame.date==pd.Timedelta(days=1)).all())

    def test_persistence_is_current_day(self):
        pred=PersistenceRegressor().fit(self.frame).predict(self.frame)
        np.testing.assert_array_equal(pred,self.frame['PM2.5'])

    def test_missing_and_corrupt_archive_raise_clear_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'missing.zip'
            with self.assertRaises(FileNotFoundError):AirQualityDataset(p).load()
            p.write_bytes(b'not a zip file')
            with self.assertRaises(DataValidationError):AirQualityDataset(p).load()

    def test_saved_metrics_recompute_from_predictions(self):
        pred=pd.read_csv(ROOT/'results/test_predictions.csv')
        scores=pd.read_csv(ROOT/'results/metrics.csv')
        for row in scores.itertuples():
            check=metrics(pred.target,pred[row.model])
            for key,value in check.items():
                self.assertAlmostEqual(getattr(row,key),value,places=8)
            self.assertTrue(np.isfinite(pred[row.model]).all())
            self.assertTrue((pred[row.model]>=0).all())

    def test_common_comparison_sample_and_coverage(self):
        pred=pd.read_csv(ROOT/'results/test_predictions.csv',parse_dates=['target_date'])
        # A full calendar test year can contain dates with no eligible station.
        self.assertEqual(pred.target_date.min(),pd.Timestamp('2016-03-01'))
        self.assertEqual(pred.target_date.max(),pd.Timestamp('2017-02-28'))
        valid=self.daily[self.daily.pm_hours>=18][['station','date']].copy()
        current=valid.assign(target_date=valid.date+pd.Timedelta(days=1))
        expected=current.merge(valid.rename(columns={'date':'target_date'}),on=['station','target_date'])
        expected=expected[expected.target_date>=TEST_START]
        self.assertEqual(set(zip(pred.station,pred.target_date)),set(zip(expected.station,expected.target_date)))
        self.assertFalse(pred.duplicated(['station','target_date']).any())
        self.assertTrue((pred.pm_hours>=18).all() and (pred.target_hours>=18).all())
        self.assertTrue((pred.target_date>=TEST_START).all())

if __name__=='__main__':unittest.main(verbosity=2)
