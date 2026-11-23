"""Rebuild the companion notebook and optionally execute it in this Python env."""
from pathlib import Path
import argparse
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# 北京多站点空气质量分析与次日 PM2.5 预测\n\n贾子硕 · 2026104376 · 2026秋《基于Python的问题解析》期末大作业\n\n本Notebook展示数据处理、模型比较和误差分析结果。默认读取随附结果；将 `RETRAIN` 设为 `True` 可重新运行实验。\n\n**核心问题：**在第t日结束时，能否仅依靠已观测历史，比“明天等于今天”更准确地预测第t+1日PM2.5？')
    code('from pathlib import Path\nimport sys, json, subprocess\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd()\nassert (ROOT / "run_analysis.py").exists(), "请将工作目录设为本项目根目录"\nRETRAIN = False\nif RETRAIN or not (ROOT / "results/metrics.csv").exists():\n    subprocess.run([sys.executable, "run_analysis.py"], cwd=ROOT, check=True)\nOUT = ROOT / "results"\ndef read_json(name):\n    return json.loads((OUT / name).read_text(encoding="utf-8"))\ndef show_fig(name):\n    display(Image(filename=str(OUT / "figures" / name), width=950))\nprint("Python:", sys.version.split()[0])\nprint("模式:", "重新运行实验" if RETRAIN else "读取已保存的结果")')
    md('## 1. 来源、原始数据与质量审计\n\n[UCI官方数据页面](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality)，DOI: [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G)，许可CC BY 4.0。原始归档不修改，只读取内层PRSA2017归档中的12个站点CSV。全样本质量审计用于发现结构问题；后续探索性图表仅使用开发期。')
    code('audit = read_json("data_audit.json")\ndisplay(pd.Series({k: audit[k] for k in ["raw_rows","raw_columns","station_count","start","end","duplicates","missing_hour_keys","daily_rows","invalid_pm_days","model_rows","development_rows","test_rows"]}, name="值").to_frame())\ndisplay(pd.DataFrame({"缺失条数":audit["missing_counts"],"缺失比例%":audit["missing_pct"]}))\nprint("SHA256:", audit["sha256"])')
    md('## 2. 类与异常处理的实际调用\n\n`AirQualityDataset` 封装读取、质量规则和聚合；`DataValidationError` 表达数据契约错误。下面刻意访问不存在的归档，捕获预期异常并打印提示，演示异常处理。输入文件缺失或损坏时，分析入口会停止并报告原因。')
    code('from src.data import AirQualityDataset, DataValidationError, build_features, calendar_folds, TEST_START\ntry:\n    AirQualityDataset(ROOT / "data/raw/example_missing.zip").load()\nexcept FileNotFoundError as exc:\n    print("预期异常已捕获:", exc)\nfinally:\n    print("示例结束；实际分析仍使用已验证的官方归档。")')
    md('## 3. 按日聚合与缺失处理\n\n日均要求至少18小时；目标不插补；当日PM2.5缺失也排除以公平比较持续性。其他特征仅在训练折内插补。风向取sin/cos日均。保留高污染值，不用统计离群规则自动删去事件。')
    code('daily = pd.read_csv(ROOT / "data/processed/daily.csv", parse_dates=["date"])\nframe, groups = build_features(daily)\ndisplay(daily.head())\nprint("特征组:", {k:len(v) for k,v in groups.items()})\nshow_fig("01_missing.png")')
    md('## 4. 分布、季节性与站点差异\n\n均值和中位数不同反映上尾；不同统计层级权重不同。图中对数变换只用于观察，不改变主模型目标。站点等权平均描述这12个监测点的浓度，未按人口分布加权。')
    code('eda = read_json("eda_summary.json")\ndisplay(pd.Series({k:v for k,v in eda.items() if not isinstance(v,dict)},name="结果").to_frame())\nshow_fig("02_distribution.png")\nshow_fig("03_time.png")')
    code('show_fig("04_stations.png")\nshow_fig("05_weather.png")')
    md('## 5. 严格的次日目标与时间验证\n\n所有特征日期≤t，目标日期=t+1。先补齐日网格再shift，防止跳过缺失日期。按目标日期统一划分全部站点，训练目标早于验证特征日期。测试区间为完整日历年，但缺失筛选后不必每天都有可评估样本。')
    code('dev = frame[frame.target_date < TEST_START - pd.Timedelta(days=1)].reset_index(drop=True)\ntest = frame[frame.target_date >= TEST_START].reset_index(drop=True)\nfolds, descriptions = calendar_folds(dev)\ndisplay(pd.DataFrame(descriptions))\nassert dev.target_date.max() < test.date.min()\nassert not set(groups["full"]) & {"target","target_date","target_hours"}\nmissing_dates = pd.date_range("2016-03-01","2017-02-28").difference(test.target_date.unique())\nprint("测试可评估日期:", test.target_date.nunique(), "日")\nprint("无可评估站点的日期:", missing_dates.strftime("%Y-%m-%d").tolist())\nprint("站点日覆盖率:", f"{len(test)/(365*12):.2%}")')
    md('## 6. 三类算法与调参\n\n岭回归用L2正则约束系数；随机森林通过多棵树平均学习非线性；HGB逐步拟合残差。均值与持续性提供参照。所有预处理在Pipeline内拟合；HGB关闭随机内部早停。共16组候选、每组3个时间折，以平均RMSE选择，不用测试集选模型。')
    code('selection = read_json("model_selection.json")\nprint("仅依据开发期CV选择的模型:", selection["selected"])\ndisplay(pd.DataFrame(selection["tuning"]).T[["params","cv_rmse","cv_mae","cv_std","candidates","seconds"]])\ncv = pd.read_csv(OUT / "cv_all.csv")\ndisplay(cv[["model","params","mean_test_rmse","mean_train_rmse","rank_test_rmse"]])\nprint("CSV采用sklearn负RMSE评分，表中取负即可获得通常的正RMSE。")\nshow_fig("06_tuning.png")')
    md('## 7. 独立测试结果\n\nRMSE用于主比较，MAE描述典型误差，R²衡量相对目标均值常数的平方误差改善，Bias识别高估或低估。主模型由开发期CV选择，独立测试年用于评价泛化。')
    code('scores = pd.read_csv(OUT / "metrics.csv").set_index("model")\ndisplay(scores.round(4))\nchosen = selection["selected"]\nimprovement = 1 - scores.loc[chosen,"RMSE"] / scores.loc["Persistence","RMSE"]\nprint(f"相对持续性的RMSE下降: {improvement:.2%}")\nshow_fig("07_models.png")')
    md('## 8. 残差、峰值和分组误差\n\n主要指标在站点日层面计算。区域平均曲线仅辅助观察，可能掩盖站点误差。高浓度阈值来自开发目标90%分位数，不是国家空气质量标准。')
    code('show_fig("08_residuals.png")\nshow_fig("09_predictions.png")')
    code('subgroups = pd.read_csv(OUT / "subgroups.csv")\ndisplay(subgroups.round(3))\nshow_fig("10_subgroups.png")\ndisplay(read_json("sensitivity.json"))')
    md('## 9. 配对块自助法与不确定性\n\n连续7日为块，所有站点随日期共同抽取，共1500次。正差值表示所选模型优于参照。区间是给定模型、给定测试期的条件评估，不是未来任何年份的性能保证。')
    code('display(pd.DataFrame(read_json("bootstrap.json")))')
    md('## 10. 置换重要性与消融\n\n单变量置换可能打破合理的特征组合，相关变量也会相互替代，因此不能解释为因果效应。消融保持算法和参数不变；额外变量并不必然提升预测。主结果不根据解释实验改选。')
    code('display(pd.read_csv(OUT / "ablation.csv").round(4))\ndisplay(pd.read_csv(OUT / "permutation_importance.csv").head(15).round(4))\nshow_fig("11_interpretation.png")')
    md('## 11. 补充稳健性与输入可获得性\n\n执行 `python extended_analysis.py` 可重算补充实验。其算法与α保持固定，评价覆盖门槛、时间块长和关键输入的影响。补充实验在初次测试后开展，用于解释与稳健性诊断。')
    code('if RETRAIN or not (OUT / "extended_summary.json").exists():\n    subprocess.run([sys.executable, "extended_analysis.py"], cwd=ROOT, check=True)\ndisplay(pd.read_csv(OUT / "feature_availability.csv").round(4))\ndisplay(pd.read_csv(OUT / "coverage_retraining.csv").round(4))\ndisplay(pd.read_csv(OUT / "coverage_common_sample.csv").round(4))\ndisplay(pd.DataFrame(read_json("block_sensitivity.json")))\nshow_fig("12_robustness.png")\nshow_fig("15_availability.png")')
    md('## 12. 高浓度事件、逐月误差与变化场景\n\n高浓度阈值固定为开发目标90%分位数。岭回归提高精确率、降低误报，同时付出召回率下降的代价。按真实次日变动分组显示，稳定日持续性更好，快速升降时岭回归降低误差。变化组用于事后诊断。')
    code('display(pd.read_csv(OUT / "high_event_diagnostics.csv").round(4))\ndisplay(read_json("extended_summary.json"))\ndisplay(pd.read_csv(OUT / "monthly_errors.csv").round(4))\ndisplay(pd.read_csv(OUT / "change_errors.csv").round(4))\nshow_fig("13_events.png")\nshow_fig("14_error_structure.png")')
    md('## 13. 可复现代码与检查\n\n主实验：`python run_analysis.py`；补充实验：`python extended_analysis.py`；完整性检查：`python -m unittest discover -s tests -v`；报告重建：`python build_report.py`。测试覆盖时间对齐、未来扰动、评价样本、逐行指标、事件计数及共同样本评分。')
    code('display(read_json("environment.json"))\nconfig = json.loads((ROOT / "project_config.json").read_text(encoding="utf-8"))\nprint("GitHub:", config["github_url"] or "未配置仓库地址")')
    md('## 14. 结论与下一步\n\n1. 覆盖门槛、训练内预处理与严格次日对齐构成可复现数据流程。\n2. 晚间污染状态提供关键预测信息，移除后平均CV RMSE上升约6.03。\n3. 岭回归总体RMSE比持续性低30.3%；平稳日持续性更好，快速升降时岭回归降低误差。\n4. 岭回归高事件精确率78.3%、召回率45.0%，提示部署需联合权衡误报与漏报。\n5. 后续可加入有真实发布时点的天气预报，开展滚动验证和分位数预测。数据与计算结果可追溯。')
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata.language_info={'name':'python','version':'3.10'}
    path=ROOT/'analysis.ipynb'
    if args.execute:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        import sys
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        NotebookClient(nb,timeout=600,km=km,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.validate(nb);nbf.write(nb,path)
    print('Notebook:',path,'code cells:',sum(c.cell_type=='code' for c in cells))

if __name__=='__main__':main()
