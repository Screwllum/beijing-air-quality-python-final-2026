"""Publication-style figures. EDA uses development data only."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

COLORS=['#176B87','#EF8B46','#6C83B5','#6A9E75','#B35B66']
CN={'Aotizhongxin':'奥体中心','Changping':'昌平','Dingling':'定陵','Dongsi':'东四',
    'Guanyuan':'官园','Gucheng':'古城','Huairou':'怀柔','Nongzhanguan':'农展馆',
    'Shunyi':'顺义','Tiantan':'天坛','Wanliu':'万柳','Wanshouxigong':'万寿西宫'}

def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,
        'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=COLORS),
        'savefig.dpi':190,'figure.dpi':120,'grid.alpha':.2})

def save(fig,path):
    fig.tight_layout(pad=1.3)
    fig.savefig(path,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def make_figures(h,d,dev,p,out,selected):
    style(); dest=out/'figures'; dest.mkdir(exist_ok=True)
    # 1: Data audit
    fields=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','wd']
    miss=100*h.groupby('station')[fields].agg(lambda s:s.isna().mean())
    fig,ax=plt.subplots(figsize=(9.2,4.4))
    im=ax.imshow(miss,cmap='YlOrRd',aspect='auto',vmin=0)
    ax.set_xticks(range(len(fields)),fields);ax.set_yticks(range(len(miss)),[CN[x] for x in miss.index])
    for i in range(len(miss)):
        for j in range(len(fields)):
            ax.text(j,i,f'{miss.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8,color='black')
    fig.colorbar(im,ax=ax,label='缺失比例 (%)',fraction=.025,pad=.02)
    ax.set_title('开发期：站点 × 变量缺失率');save(fig,dest/'01_missing.png')
    # 2: distribution and tail
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5))
    vals=d['PM2.5'].dropna()
    axs[0].hist(vals,bins=55,color=COLORS[0],alpha=.85)
    axs[0].axvline(vals.median(),color=COLORS[1],ls='--',label=f'中位数 {vals.median():.1f}')
    axs[0].set(xlabel='日均 PM2.5 (μg/m³)',ylabel='站点日数量',title='日均浓度呈右偏分布');axs[0].legend()
    axs[1].hist(np.log1p(vals),bins=45,color=COLORS[2],alpha=.85)
    axs[1].set(xlabel='log(1 + 日均 PM2.5)',ylabel='站点日数量',title='对数尺度仅用于观察分布')
    save(fig,dest/'02_distribution.png')
    # 3 time seasonal
    fig,axs=plt.subplots(2,1,figsize=(9.2,5.7),gridspec_kw={'height_ratios':[1.25,1]})
    city=d.groupby('date')['PM2.5'].agg(['mean','count'])
    series=city['mean'].where(city['count']>=8)
    axs[0].plot(series.index,series,lw=.55,alpha=.4,label='各站点日均值的等权平均')
    axs[0].plot(series.index,series.rolling(30,min_periods=20).mean(),lw=1.8,color=COLORS[1],label='30日后向滚动平均')
    axs[0].set(ylabel='PM2.5 (μg/m³)',title='开发期的区域共同波动');axs[0].legend(fontsize=8,ncol=2)
    months=d.groupby(d.date.dt.month)['PM2.5'].agg(['mean','median'])
    axs[1].plot(months.index,months['mean'],'o-',label='均值');axs[1].plot(months.index,months['median'],'s--',label='中位数')
    axs[1].set(xticks=range(1,13),xlabel='月份',ylabel='PM2.5 (μg/m³)',title='按月份汇总的季节性（含不同年份）');axs[1].legend(fontsize=8)
    save(fig,dest/'03_time.png')
    # 4 site matrix and means
    fig,axs=plt.subplots(1,2,figsize=(9.2,4.3),gridspec_kw={'width_ratios':[1,1.3]})
    means=d.groupby('station')['PM2.5'].mean().sort_values()
    axs[0].barh([CN[s] for s in means.index],means.values,color=COLORS[0]);axs[0].set(xlabel='日均 PM2.5 (μg/m³)',title='站点平均浓度')
    piv=d.pivot(index='date',columns='station',values='PM2.5').corr(method='spearman')
    im=axs[1].imshow(piv,vmin=.5,vmax=1,cmap='Blues')
    axs[1].set_xticks(range(12),[CN[s] for s in piv.columns],rotation=60,ha='right',fontsize=8)
    axs[1].set_yticks(range(12),[CN[s] for s in piv.index],fontsize=8);axs[1].set_title('站点日均浓度 Spearman 相关')
    fig.colorbar(im,ax=axs[1],fraction=.045,pad=.03);save(fig,dest/'04_stations.png')
    # 5 weather and other pollutants
    fig,axs=plt.subplots(1,2,figsize=(9.2,4.2))
    cols=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','WSPM']
    corr=d[cols].corr(method='spearman')
    im=axs[0].imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
    axs[0].set_xticks(range(len(cols)),cols,rotation=60,ha='right',fontsize=8);axs[0].set_yticks(range(len(cols)),cols,fontsize=8)
    axs[0].set_title('日均变量的 Spearman 相关');fig.colorbar(im,ax=axs[0],fraction=.045,pad=.03)
    bins=pd.cut(d.WSPM,[0,1,2,3,4,20],include_lowest=True)
    groups=[d.loc[bins==c,'PM2.5'].dropna() for c in bins.cat.categories]
    axs[1].boxplot(groups,tick_labels=['≤1','1-2','2-3','3-4','>4'],showfliers=False,patch_artist=True,
        boxprops={'facecolor':'#D8E8EF'})
    axs[1].set(xlabel='日均风速 (m/s)',ylabel='日均 PM2.5 (μg/m³)',title='风速分组（隐藏离群点，仅为绘图）')
    save(fig,dest/'05_weather.png')
    # 6 cv curves
    cv=pd.read_csv(out/'cv_all.csv'); fig,axs=plt.subplots(1,3,figsize=(9.2,3.6))
    for ax,(name,c) in zip(axs,cv.groupby('model',sort=False)):
        xs=np.arange(1,len(c)+1)
        ax.errorbar(xs,-c.mean_test_rmse,yerr=c.std_test_rmse,fmt='o-',capsize=3,label='验证 ± 折间标准差')
        ax.plot(xs,-c.mean_train_rmse,'s--',alpha=.65,label='训练')
        ax.set(title=name,xlabel='候选组合序号',ylabel='RMSE (μg/m³)');ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y');ax.legend(fontsize=7)
    save(fig,dest/'06_tuning.png')
    # 7 models
    scores=pd.read_csv(out/'metrics.csv').sort_values('RMSE')
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.6))
    names=scores.model.str.replace('HistGradientBoosting','HGB').str.replace('RandomForest','RF')
    for ax,metric in zip(axs,['RMSE','MAE']):
        bars=ax.barh(names,scores[metric],color=[COLORS[1] if x==selected else COLORS[0] for x in scores.model])
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9);ax.set_xlim(0,scores[metric].max()*1.18)
        ax.set(xlabel=f'{metric} (μg/m³)',title=f'独立测试年：{metric}');ax.invert_yaxis()
    save(fig,dest/'07_models.png')
    # 8 diagnostic
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.7))
    hb=axs[0].hexbin(p.target,p[selected],gridsize=35,mincnt=1,cmap='Blues',bins='log')
    lim=max(p.target.max(),p[selected].max());axs[0].plot([0,lim],[0,lim],'--',color=COLORS[1])
    axs[0].set(xlabel='实际次日日均 PM2.5',ylabel='预测 PM2.5',title='预测值与实际值 (μg/m³)')
    fig.colorbar(hb,ax=axs[0],label='格内样本数',fraction=.04)
    residual=p[selected]-p.target
    axs[1].scatter(p.target,residual,s=4,alpha=.18,color=COLORS[0]);axs[1].axhline(0,color=COLORS[1],ls='--')
    axs[1].set(xlabel='实际次日日均 PM2.5',ylabel='预测 - 实际 (μg/m³)',title='高浓度日是否存在系统低估？')
    save(fig,dest/'08_residuals.png')
    # 9 full-year and fixed window (not cherry-picked)
    cityp=p.groupby('target_date')[['target',selected,'Persistence']].mean()
    fig,axs=plt.subplots(2,1,figsize=(9.2,5.2))
    for ax,subset,title in [(axs[0],cityp,'整个测试年：各站点的等权日平均'),
                           (axs[1],cityp.loc['2016-12-01':'2017-01-31'],'预先指定的冬季窗口：2016年12月-2017年1月')]:
        ax.plot(subset.index,subset.target,label='实际',lw=1.2)
        ax.plot(subset.index,subset[selected],label=selected,lw=1.1)
        ax.set(ylabel='PM2.5 (μg/m³)',title=title);ax.legend(fontsize=8)
    save(fig,dest/'09_predictions.png')
    # 10 error strata
    sub=pd.read_csv(out/'subgroups.csv'); fig,axs=plt.subplots(1,2,figsize=(9.2,3.6))
    for ax,key in zip(axs,['season','level']):
        z=sub[sub.dimension==key];x=np.arange(len(z));w=.34
        ax.bar(x-w/2,z.RMSE,w,label=selected);ax.bar(x+w/2,z.persistence_RMSE,w,label='Persistence')
        ax.set(xticks=x,xticklabels=z.group,ylabel='RMSE (μg/m³)',title='按季节' if key=='season' else '按开发期90%分位阈值分组')
        ax.legend(fontsize=8)
    save(fig,dest/'10_subgroups.png')
    # 11 ablation and importance
    imp=pd.read_csv(out/'permutation_importance.csv').head(10).iloc[::-1]
    abl=pd.read_csv(out/'ablation.csv')
    fig,axs=plt.subplots(1,2,figsize=(9.2,4.2))
    axs[0].barh(imp.feature,imp.delta_rmse,xerr=imp['std'],color=COLORS[0]);axs[0].set(xlabel='置换后 RMSE 增量 (μg/m³)',title='最后一折验证集：前10项置换重要性')
    bars=axs[1].bar(['历史+日历','再加气象','全部特征'],abl.cv_rmse,color=COLORS[:3]);axs[1].bar_label(bars,fmt='%.2f',padding=3)
    axs[1].set(ylabel='三折平均 RMSE (μg/m³)',title='固定算法和参数的特征消融');axs[1].set_ylim(0,abl.cv_rmse.max()*1.2)
    save(fig,dest/'11_interpretation.png')
