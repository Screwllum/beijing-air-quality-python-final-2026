"""One command reproduces all results: python run_analysis.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json
import platform
import importlib.metadata
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from src.data import AirQualityDataset, build_features, calendar_folds, TEST_START
from src.modeling import PersistenceRegressor, tune_models, metrics, block_bootstrap, make_pipeline, model_specs

ROOT=Path(__file__).resolve().parent

def save_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def main():
    started=time.perf_counter()
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    processed=ROOT/'data'/'processed'; processed.mkdir(exist_ok=True)
    data=AirQualityDataset(ROOT/'data'/'raw'/'beijing-air-quality.zip')
    hourly=data.load(); daily=data.aggregate(hourly)
    frame, groups=build_features(daily)
    # One-day purge between the last training target and first test feature date.
    dev=frame[frame.target_date < TEST_START-pd.Timedelta(days=1)].reset_index(drop=True)
    test=frame[frame.target_date >= TEST_START].reset_index(drop=True)
    folds,fold_desc=calendar_folds(dev)
    features=groups['full']
    assert set(features).isdisjoint({'target','target_date','target_hours'})
    assert dev.target_date.max() < test.date.min()
    daily.to_csv(processed/'daily.csv',index=False,encoding='utf-8-sig')
    frame.to_csv(processed/'model_table.csv',index=False,encoding='utf-8-sig')
    data.audit.update(model_rows=len(frame),development_rows=len(dev),test_rows=len(test),
                      features=features,feature_count=len(features),folds=fold_desc,
                      dev_target_start=str(dev.target_date.min().date()),
                      dev_target_end=str(dev.target_date.max().date()),
                      test_target_start=str(test.target_date.min().date()),
                      test_target_end=str(test.target_date.max().date()),
                      test_observed_days=test.target_date.nunique(),
                      test_calendar_days=len(pd.date_range(TEST_START,test.target_date.max())),
                      test_missing_dates=pd.date_range(TEST_START,test.target_date.max()).difference(
                          test.target_date.unique()).strftime('%Y-%m-%d').tolist(),
                      purged_rows=int(((frame.target_date>=TEST_START-pd.Timedelta(days=1)) & (frame.target_date<TEST_START)).sum()))
    save_json(out/'data_audit.json',data.audit)
    # EDA is restricted to the development period, before test predictions are examined.
    hdev=hourly[hourly.timestamp < TEST_START-pd.Timedelta(days=1)].copy()
    ddev=daily[daily.date < TEST_START-pd.Timedelta(days=1)].copy()
    hdev.groupby('station')[['PM2.5','PM10','NO2','O3']].mean().to_csv(out/'station_hourly_means.csv')
    ddev.describe(include='all').to_csv(out/'daily_descriptive.csv')
    eda={'dev_hourly_pm_mean':float(hdev['PM2.5'].mean()),
         'dev_hourly_pm_median':float(hdev['PM2.5'].median()),
         'dev_hourly_pm_p95':float(hdev['PM2.5'].quantile(.95)),
         'dev_hourly_pm_max':float(hdev['PM2.5'].max()),
         'dev_daily_pm_mean':float(ddev['PM2.5'].mean()),
         'dev_pm_missing_pct':float(100*hdev['PM2.5'].isna().mean()),
         'pm_pm10_spearman':float(ddev[['PM2.5','PM10']].corr(method='spearman').iloc[0,1]),
         'pm_wind_spearman':float(ddev[['PM2.5','WSPM']].corr(method='spearman').iloc[0,1]),
         'pm_persistence_spearman':float(dev[['PM2.5','target']].corr(method='spearman').iloc[0,1]),
         'station_daily_mean':ddev.groupby('station')['PM2.5'].mean().to_dict(),
         'month_daily_mean':ddev.groupby(ddev.date.dt.month)['PM2.5'].mean().to_dict(),
         'dev_target_p90':float(dev.target.quantile(.9))}
    save_json(out/'eda_summary.json',eda)
    with threadpool_limits(limits=4):
        models, tuning=tune_models(dev,features,folds,out)
        selected=min(tuning,key=lambda n:tuning[n]['cv_rmse'])
        # Freeze winner before looking at the independent test set.
        save_json(out/'model_selection.json',{'criterion':'minimum mean calendar-CV RMSE','selected':selected,'tuning':tuning})
        print('Selected before testing:',selected,flush=True)
        # Baselines use the identical folds and evaluation sample.
        baseline_cv={}
        baselines={'TrainingMean':DummyRegressor(strategy='mean'),'Persistence':PersistenceRegressor()}
        for name,model in baselines.items():
            scores=[]
            for tr,va in folds:
                fitted=clone(model).fit(dev.iloc[tr][features],dev.iloc[tr].target)
                scores.append(metrics(dev.iloc[va].target,fitted.predict(dev.iloc[va][features]))['RMSE'])
            baseline_cv[name]=scores
            models[name]=model.fit(dev[features],dev.target)
        # Pre-declared ablation: fixed winning algorithm/parameters, no retuning.
        ablations=[]
        for group,cols in groups.items():
            alg=clone(models[selected].estimator_.named_steps['model'])
            pipe=make_pipeline(alg,cols)
            scores=[]
            for tr,va in folds:
                model=clone(pipe).fit(dev.iloc[tr][cols],dev.iloc[tr].target)
                scores.append(metrics(dev.iloc[va].target,model.predict(dev.iloc[va][cols]))['RMSE'])
            ablations.append({'features':group,'n_features':len(cols),'cv_rmse':float(np.mean(scores)),
                              **{f'fold{i+1}':s for i,s in enumerate(scores)}})
        pd.DataFrame(ablations).to_csv(out/'ablation.csv',index=False)
        # Validation-only interpretation; does not drive further selection.
        tr,va=folds[-1]
        interp=clone(models[selected]).fit(dev.iloc[tr][features],dev.iloc[tr].target)
        imp=permutation_importance(interp,dev.iloc[va][features],dev.iloc[va].target,
            scoring='neg_root_mean_squared_error',n_repeats=5,random_state=42,n_jobs=1)
        pd.DataFrame({'feature':features,'delta_rmse':imp.importances_mean,'std':imp.importances_std}).sort_values(
            'delta_rmse',ascending=False).to_csv(out/'permutation_importance.csv',index=False)
        predictions=test[['station','date','target_date','target','pm_hours','target_hours']].copy()
        score_rows=[]
        for name,model in models.items():
            t=time.perf_counter(); pred=model.predict(test[features]); seconds=time.perf_counter()-t
            predictions[name]=pred
            cv=tuning[name]['cv_rmse'] if name in tuning else float(np.mean(baseline_cv[name]))
            score_rows.append({'model':name,'CV_RMSE':cv,**metrics(test.target,pred),'predict_seconds':seconds})
    scores=pd.DataFrame(score_rows).sort_values('CV_RMSE')
    scores.to_csv(out/'metrics.csv',index=False)
    predictions.to_csv(out/'test_predictions.csv',index=False,encoding='utf-8-sig')
    bootstrap=[block_bootstrap(predictions,selected,ref) for ref in ['Persistence','RandomForest','Ridge'] if ref!=selected]
    save_json(out/'bootstrap.json',bootstrap)
    # Error groups use target-day season; q90 is fixed using training data only.
    predictions['season']=predictions.target_date.dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'})
    predictions['level']=np.where(predictions.target>=eda['dev_target_p90'],'High (train q90)','Other')
    group_rows=[]
    for key in ['season','station','level']:
        for group,p in predictions.groupby(key):
            group_rows.append({'dimension':key,'group':group,'n':len(p),**metrics(p.target,p[selected]),
                               'persistence_RMSE':metrics(p.target,p.Persistence)['RMSE']})
    pd.DataFrame(group_rows).to_csv(out/'subgroups.csv',index=False)
    strict=predictions[(predictions.pm_hours>=20)&(predictions.target_hours>=20)]
    save_json(out/'sensitivity.json',{'description':'same fitted models; both days have >=20 valid PM hours',
        'n':len(strict),'selected':metrics(strict.target,strict[selected]),'persistence':metrics(strict.target,strict.Persistence)})
    env={'python':platform.python_version(), 'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','scipy','scikit-learn','reportlab','pypdf','joblib','threadpoolctl']},
        'seed':42,'elapsed_seconds':time.perf_counter()-started}
    save_json(out/'environment.json',env)
    from src.plots import make_figures
    make_figures(hdev,ddev,dev,predictions,out,selected)
    print(scores.to_string(index=False),flush=True)
    print(f'Done in {time.perf_counter()-started:.1f}s',flush=True)

if __name__=='__main__':
    try:
        main()
    except (FileNotFoundError,ValueError) as exc:
        raise SystemExit(f'Analysis stopped: {exc}. Check data archive and schema.') from exc
