"""Verify supplementary experiments from their saved row-level evidence."""
from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.modeling import metrics

ROOT=Path(__file__).resolve().parents[1]

class SupplementaryChecks(unittest.TestCase):
    def test_event_counts_recomputed_from_forecasts(self):
        p=pd.read_csv(ROOT/'results/test_predictions.csv')
        e=pd.read_csv(ROOT/'results/high_event_diagnostics.csv')
        for row in e.itertuples():
            counts=confusion_matrix(p.target>=row.threshold,p[row.model]>=row.threshold,labels=[False,True]).ravel()
            np.testing.assert_array_equal(counts,[row.tn,row.fp,row.fn,row.tp])
            self.assertEqual(int(sum(counts)),len(p))
            self.assertAlmostEqual(row.recall,row.tp/(row.tp+row.fn))

    def test_coverage_results_use_matching_common_population(self):
        predictions=pd.read_csv(ROOT/'results/coverage_predictions.csv',parse_dates=['target_date'])
        summary=pd.read_csv(ROOT/'results/coverage_retraining.csv')
        common=pd.read_csv(ROOT/'results/coverage_common_sample.csv')
        groups=[g for _,g in predictions.groupby('min_hours')]
        keys=set.intersection(*(set(zip(p.station,p.target_date)) for p in groups))
        for p in groups:
            threshold=int(p.min_hours.iloc[0]);row=summary[summary.min_hours==threshold].iloc[0]
            self.assertEqual(len(p),row.n_test)
            self.assertAlmostEqual(metrics(p.target,p.prediction)['RMSE'],row.RMSE)
            paired=p[[key in keys for key in zip(p.station,p.target_date)]]
            row=common[common.min_hours==threshold].iloc[0]
            self.assertEqual(len(paired),row.n_common)
            self.assertAlmostEqual(metrics(paired.target,paired.prediction)['RMSE'],row.RMSE)

    def test_block_sensitivity_keeps_same_point_difference(self):
        b=json.loads((ROOT/'results/block_sensitivity.json').read_text())
        self.assertEqual(len(b),10)
        for ref in ['Persistence','RandomForest']:
            values=[x for x in b if x['reference']==ref]
            np.testing.assert_allclose([x['rmse_improvement'] for x in values],values[0]['rmse_improvement'])
            self.assertEqual({x['block_days'] for x in values},{1,3,7,14,28})
            self.assertTrue(all(np.isfinite(x['ci95']).all() and x['ci95'][0]<=x['ci95'][1] for x in values))

    def test_monthly_and_change_groups_partition_test_samples(self):
        n=len(pd.read_csv(ROOT/'results/test_predictions.csv'))
        for file in ['monthly_errors.csv','change_errors.csv']:
            self.assertEqual(int(pd.read_csv(ROOT/'results'/file).n.sum()),n)

if __name__=='__main__':unittest.main()
