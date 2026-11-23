"""Models, time validation, nonnegative predictions and block uncertainty."""
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, make_scorer

class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Tomorrow equals today's observed mean; no future information."""
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        return X['PM2.5'].to_numpy(dtype=float)

class NonnegativeRegressor(RegressorMixin, BaseEstimator):
    """Apply the same physical prediction constraint during CV and testing."""
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X,y)
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        return np.maximum(0, self.estimator_.predict(X))

def make_pipeline(model, features):
    cats = [c for c in ['station','weekday'] if c in features]
    nums = [c for c in features if c not in cats]
    prep = ColumnTransformer([
        ('num', Pipeline([('impute',SimpleImputer(strategy='median', add_indicator=True)),
                          ('scale',StandardScaler())]), nums),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)],
        verbose_feature_names_out=False)
    return NonnegativeRegressor(Pipeline([('preprocess',prep),('model',model)]))

def model_specs(features):
    prefix = 'estimator__model__'
    return {
        'Ridge': (make_pipeline(Ridge(),features), {prefix+'alpha':[0.1,1,10,100]}),
        'RandomForest': (make_pipeline(RandomForestRegressor(n_estimators=160,
            random_state=42,n_jobs=4),features),
            {prefix+'max_depth':[10,None],prefix+'min_samples_leaf':[3,10],prefix+'max_features':[0.8]}),
        'HistGradientBoosting': (make_pipeline(HistGradientBoostingRegressor(
            max_iter=240,early_stopping=False,random_state=42),features),
            {prefix+'learning_rate':[0.05,0.1],prefix+'max_leaf_nodes':[15,31],prefix+'l2_regularization':[0,10]})
    }

def metrics(y, pred):
    return {'MAE':float(mean_absolute_error(y,pred)),
            'RMSE':float(root_mean_squared_error(y,pred)),
            'R2':float(r2_score(y,pred)),
            'Bias':float(np.mean(np.asarray(pred)-np.asarray(y)))}

def tune_models(dev, features, folds, output):
    models, summaries, details = {}, {}, []
    X,y = dev[features], dev.target
    for name,(pipe,grid) in model_specs(features).items():
        print(f'Tuning {name} ...', flush=True)
        start = time.perf_counter()
        search = GridSearchCV(pipe,grid,cv=folds,scoring={
            'rmse':'neg_root_mean_squared_error','mae':'neg_mean_absolute_error'},
            refit='rmse',n_jobs=1,return_train_score=True,error_score='raise')
        search.fit(X,y)
        frame = pd.DataFrame(search.cv_results_)
        frame.insert(0,'model',name)
        frame.to_csv(output/f'cv_{name}.csv',index=False,encoding='utf-8-sig')
        idx = search.best_index_
        summaries[name] = {'params':search.best_params_,
            'cv_rmse':float(-search.best_score_),
            'cv_std':float(frame.loc[idx,'std_test_rmse']),
            'cv_mae':float(-frame.loc[idx,'mean_test_mae']),
            'fold_rmse':[float(-frame.loc[idx,f'split{i}_test_rmse']) for i in range(len(folds))],
            'candidates':len(frame),'seconds':time.perf_counter()-start,
            'default_grid_first_rmse':float(-frame.iloc[0].mean_test_rmse)}
        models[name]=search.best_estimator_
        details.append(frame)
        print(name, summaries[name], flush=True)
    pd.concat(details,ignore_index=True).to_csv(output/'cv_all.csv',index=False,encoding='utf-8-sig')
    return models,summaries

def block_bootstrap(predictions, chosen, reference='Persistence', repeats=1500, block=7):
    """Resample contiguous calendar-day blocks, keeping all stations together.

    Difference = RMSE(reference) - RMSE(chosen); positive favors chosen.
    This estimates conditional evaluation uncertainty, not retraining uncertainty.
    """
    p=predictions.copy()
    p['sq_model']=(p[chosen]-p.target)**2
    p['sq_ref']=(p[reference]-p.target)**2
    daily=p.groupby('target_date').agg(sq_model=('sq_model','sum'),sq_ref=('sq_ref','sum'),n=('target','size'))
    dates=pd.date_range(daily.index.min(),daily.index.max(),freq='D')
    arr=daily.reindex(dates,fill_value=0).to_numpy()
    rng=np.random.default_rng(42)
    draws=[]
    for _ in range(repeats):
        starts=rng.integers(0,len(arr)-block+1,size=int(np.ceil(len(arr)/block)))
        idx=np.concatenate([np.arange(s,s+block) for s in starts])[:len(arr)]
        s=arr[idx].sum(axis=0)
        draws.append(np.sqrt(s[1]/s[2])-np.sqrt(s[0]/s[2]))
    original=np.sqrt(arr[:,1].sum()/arr[:,2].sum())-np.sqrt(arr[:,0].sum()/arr[:,2].sum())
    return {'model':chosen,'reference':reference,'block_days':block,'repeats':repeats,
        'rmse_improvement':float(original),'ci95':np.quantile(draws,[0.025,0.975]).tolist()}
