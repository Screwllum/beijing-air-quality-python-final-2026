"""Supplementary robustness and event diagnostics, keeping the main model fixed.

Run after run_analysis.py. No test result is used to change model selection.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from threadpoolctl import threadpool_limits
from src.data import AirQualityDataset, build_features, calendar_folds, TEST_START
from src.modeling import make_pipeline, metrics, block_bootstrap
from src.plots import style, save, COLORS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def run():
    frame=pd.read_csv(ROOT/'data/processed/model_table.csv',parse_dates=['date','target_date'])
    daily=pd.read_csv(ROOT/'data/processed/daily.csv',parse_dates=['date'])
    _,groups=build_features(daily)
    dev=frame[frame.target_date<TEST_START-pd.Timedelta(days=1)].reset_index(drop=True)
    test=frame[frame.target_date>=TEST_START].copy()
    pred=pd.read_csv(OUT/'test_predictions.csv',parse_dates=['date','target_date'])
    folds,_=calendar_folds(dev)
    params=json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
    assert params['selected']=='Ridge'
    alpha=params['tuning']['Ridge']['params']['estimator__model__alpha']
    threshold=float(dev.target.quantile(.9))
    feature_sets={'full':groups['full'],
        'no_evening':[x for x in groups['full'] if x!='pm_evening'],
        'no_city':[x for x in groups['full'] if x!='city_pm'],
        'no_evening_or_city':[x for x in groups['full'] if x not in ['pm_evening','city_pm']]}
    timing=[]
    with threadpool_limits(limits=4):
        for name,features in feature_sets.items():
            scores=[]
            for tr,va in folds:
                model=make_pipeline(Ridge(alpha=alpha),features).fit(dev.iloc[tr][features],dev.iloc[tr].target)
                scores.append(metrics(dev.iloc[va].target,model.predict(dev.iloc[va][features]))['RMSE'])
            timing.append({'features':name,'n_features':len(features),'CV_RMSE':np.mean(scores),
                           **{f'fold{i+1}':s for i,s in enumerate(scores)}})
        pd.DataFrame(timing).to_csv(OUT/'feature_availability.csv',index=False)
        data=AirQualityDataset(ROOT/'data/raw/beijing-air-quality.zip')
        hourly=data.load()
        thresholds=[]; predictions=[]
        for min_hours in [18,20,24]:
            data=AirQualityDataset(data.archive,min_hours=min_hours)
            df,grp=build_features(data.aggregate(hourly))
            train=df[df.target_date<TEST_START-pd.Timedelta(days=1)]
            ts=df[df.target_date>=TEST_START]
            model=make_pipeline(Ridge(alpha=alpha),grp['full']).fit(train[grp['full']],train.target)
            p=ts[['station','target_date','target','PM2.5']].copy()
            p['prediction']=model.predict(ts[grp['full']]);p['min_hours']=min_hours
            predictions.append(p)
            thresholds.append({'min_hours':min_hours,'n_train':len(train),'n_test':len(ts),
                               'coverage_pct':100*len(ts)/(365*12),
                               **metrics(ts.target,p.prediction),
                               'persistence_RMSE':metrics(ts.target,ts['PM2.5'])['RMSE']})
    # A common evaluation population separates coverage filtering from model fitting.
    shared=set.intersection(*(set(zip(p.station,p.target_date)) for p in predictions))
    common=[]
    for p in predictions:
        p=p[[k in shared for k in zip(p.station,p.target_date)]]
        common.append({'min_hours':int(p.min_hours.iloc[0]),'n_common':len(p),**metrics(p.target,p.prediction)})
    pd.DataFrame(thresholds).to_csv(OUT/'coverage_retraining.csv',index=False)
    pd.DataFrame(common).to_csv(OUT/'coverage_common_sample.csv',index=False)
    pd.concat(predictions).to_csv(OUT/'coverage_predictions.csv',index=False)
    blocks=[]
    for block in [1,3,7,14,28]:
        for reference in ['Persistence','RandomForest']:
            blocks.append(block_bootstrap(pred,'Ridge',reference,repeats=1500,block=block))
    (OUT/'block_sensitivity.json').write_text(json.dumps(blocks,indent=2),encoding='utf-8')
    events=[]
    truth=pred.target>=threshold
    for name in ['Persistence','Ridge','RandomForest','HistGradientBoosting']:
        forecast=pred[name]>=threshold
        tn,fp,fn,tp=confusion_matrix(truth,forecast,labels=[False,True]).ravel()
        events.append({'model':name,'threshold':threshold,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'precision':precision_score(truth,forecast,zero_division=0),
            'recall':recall_score(truth,forecast,zero_division=0),
            'f1':f1_score(truth,forecast,zero_division=0),
            'n_alerts':int(forecast.sum())})
    pd.DataFrame(events).to_csv(OUT/'high_event_diagnostics.csv',index=False)
    pred['month']=pred.target_date.dt.strftime('%Y-%m')
    monthly=[]
    for month,p in pred.groupby('month'):
        monthly.append({'month':month,'n':len(p),'target_mean':p.target.mean(),
            **metrics(p.target,p.Ridge),'persistence_RMSE':metrics(p.target,p.Persistence)['RMSE']})
    pd.DataFrame(monthly).to_csv(OUT/'monthly_errors.csv',index=False)
    # Same-day increase/decrease uses observed current mean joined by station/date.
    p=pred.merge(test[['station','target_date','PM2.5']],on=['station','target_date'],validate='one_to_one')
    p['change_group']=np.select([p.target-p['PM2.5']>=50,p.target-p['PM2.5']<=-50],
                                ['Increase >=50','Decrease >=50'],default='Change <50')
    changes=[]
    for group,z in p.groupby('change_group'):
        changes.append({'group':group,'n':len(z),**metrics(z.target,z.Ridge),
                        'persistence_RMSE':metrics(z.target,z.Persistence)['RMSE']})
    pd.DataFrame(changes).to_csv(OUT/'change_errors.csv',index=False)
    # A high-event error budget shows why a small tail matters to overall RMSE.
    sq=(pred.Ridge-pred.target)**2
    summary={'threshold':threshold,'high_n':int(truth.sum()),'high_fraction':float(truth.mean()),
        'high_sse_fraction':float(sq[truth].sum()/sq.sum()),
        'method':'Supplementary diagnostics added after the initial test evaluation; main model/parameters unchanged.',
        'alpha':alpha,'bootstrap_seed':42,'bootstrap_repeats':1500}
    (OUT/'extended_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    make_plots(timing,thresholds,common,blocks,events,monthly,changes,threshold)
    print(pd.DataFrame(thresholds).to_string(index=False))
    print(pd.DataFrame(events).to_string(index=False))
    print(summary)

def make_plots(timing,thresholds,common,blocks,events,monthly,changes,threshold):
    style();dest=OUT/'figures'
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.8))
    for ref,offset,color in [('Persistence',-.08,COLORS[0]),('RandomForest',.08,COLORS[1])]:
        a=[b for b in blocks if b['reference']==ref];x=np.arange(len(a))+offset
        y=np.array([b['rmse_improvement'] for b in a]);lo=np.array([b['ci95'][0] for b in a]);hi=np.array([b['ci95'][1] for b in a])
        axs[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=4,color=color,label=ref)
    axs[0].axhline(0,color='gray',ls='--');axs[0].set(xticks=range(5),xticklabels=[1,3,7,14,28],
        xlabel='连续抽样块长（日）',ylabel='参照RMSE - Ridge RMSE',title='改进区间对时间块长的敏感性');axs[0].legend(fontsize=8)
    a=pd.DataFrame(thresholds);c=pd.DataFrame(common)
    axs[1].plot(a.min_hours,a.RMSE,'o-',label='各自合格样本')
    axs[1].plot(c.min_hours,c.RMSE,'s--',label=f'共同样本 n={c.n_common.iloc[0]}')
    axs[1].set(xticks=[18,20,24],xlabel='当日/次日的有效小时门槛',ylabel='重新训练 Ridge 的 RMSE',title='重新聚合、构造特征与训练');axs[1].legend(fontsize=8)
    save(fig,dest/'12_robustness.png')
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.6))
    e=pd.DataFrame(events);xx=np.arange(len(e))
    for metric,offset,col in [('precision',-.23,COLORS[0]),('recall',0,COLORS[1]),('f1',.23,COLORS[2])]:
        axs[0].bar(xx+offset,e[metric],width=.23,color=col,label=metric)
    axs[0].set(xticks=xx,xticklabels=['持续性','Ridge','RF','HGB'],ylim=(0,1),ylabel='比例',title=f'高浓度事件诊断：阈值 {threshold:.2f}')
    axs[0].legend(fontsize=8)
    row=e[e.model=='Ridge'].iloc[0];matrix=np.array([[row.tn,row.fp],[row.fn,row.tp]])
    axs[1].imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):axs[1].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=15,color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    axs[1].set(xticks=[0,1],xticklabels=['预测普通','预测高浓度'],yticks=[0,1],yticklabels=['实际普通','实际高浓度'],title='Ridge：4,214个站点日的混淆矩阵')
    save(fig,dest/'13_events.png')
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.7),gridspec_kw={'width_ratios':[1.4,1]})
    m=pd.DataFrame(monthly);x=np.arange(len(m))
    axs[0].plot(x,m.RMSE,'o-',label='Ridge');axs[0].plot(x,m.persistence_RMSE,'s--',label='持续性')
    axs[0].set(xticks=x,xticklabels=m.month.str.slice(2),ylabel='RMSE (μg/m³)',title='测试年逐月误差');axs[0].tick_params(axis='x',rotation=55);axs[0].legend(fontsize=8)
    z=pd.DataFrame(changes);xx=np.arange(len(z))
    axs[1].bar(xx-.17,z.RMSE,width=.34,label='Ridge');axs[1].bar(xx+.17,z.persistence_RMSE,width=.34,label='持续性')
    axs[1].set(xticks=xx,xticklabels=['变动<50','下降≥50','上升≥50'],ylabel='RMSE (μg/m³)',title='按次日真实变化幅度诊断');axs[1].legend(fontsize=8)
    save(fig,dest/'14_error_structure.png')
    fig,ax=plt.subplots(figsize=(9.2,2.8))
    z=pd.DataFrame(timing);bars=ax.barh(['全部特征','移除晚间PM2.5','移除区域均值','同时移除两者'],z.CV_RMSE,color=COLORS[:4])
    ax.bar_label(bars,fmt='%.2f',padding=4);ax.set(xlabel='三折平均 RMSE (μg/m³)',xlim=(0,z.CV_RMSE.max()*1.14),title='数据可获得性：固定 Ridge α=10 的开发期验证');ax.invert_yaxis()
    save(fig,dest/'15_availability.png')

if __name__=='__main__':run()
