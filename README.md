# 北京多站点空气质量分析与次日 PM2.5 预测

2026 秋《基于 Python 的问题解析》期末大作业。贾子硕，2026104376。

项目仓库：https://github.com/Screwllum/beijing-air-quality-python-final-2026

[阅读15页PDF报告](report/贾子硕2026104376期末大作业.pdf) · [打开已执行Notebook](analysis.ipynb)

本项目使用公开的12站点小时数据，完成质量审计、日聚合、探索分析、时间特征工程、机器学习调参、独立年度评价及事件诊断。在第t日结束时，以截至当天的污染、气象和历史信息，预测同一站点第t+1日平均PM2.5。

## 研究问题

1. 缺失值、重复记录、异常值和时间连续性如何影响分析？
2. PM2.5 的分布、月度变化、站点差异和共同波动有什么规律？
3. 气象和其他污染物与 PM2.5 有哪些关联？哪些不能解释为因果？
4. 机器学习能否超过“明天等于今天”的持续性预测？
5. 哪些季节、站点、高污染日更难预测，模型改进是否稳定？

## 模型结果

| 方法 | 时间CV RMSE | 测试MAE | 测试RMSE | 测试R² |
|---|---:|---:|---:|---:|
| Ridge | 45.71 | 31.87 | 46.18 | 0.591 |
| RandomForest | 46.29 | 33.26 | 51.72 | 0.487 |
| HistGradientBoosting | 48.98 | 34.98 | 54.27 | 0.435 |
| Persistence | 61.06 | 44.48 | 66.27 | 0.158 |
| TrainingMean | 68.85 | 51.82 | 72.23 | -0.001 |

误差单位为μg/m³。岭回归由开发期CV选中，相对持续性测试RMSE降低约30.3%。测试日历跨度365日，质量筛选后有4,214个站点日、363个可评估日期；2016-09-07和2016-09-08缺少合格站点对。完整精度见 `results/metrics.csv`。

补充分析结果：

- 晚间PM2.5是关键输入：移除后平均CV RMSE从45.71升至51.75。
- 覆盖门槛18/20/24小时均重新聚合、构造特征和训练；共同的3,187个样本上RMSE分别为47.71/47.69/47.98。
- 1/3/7/14/28日块自助抽样中，岭回归相对持续性和随机森林的RMSE改善区间均保持正值。
- 岭回归高浓度事件精确率78.3%、召回率45.0%、F1为0.572；相对持续性减少152次误报、增加34次漏报。
- 日均绝对变化<50时持续性更好；快速升降时岭回归降低误差。
- 高浓度组占10.28%的样本，贡献47.76%的岭回归平方误差。

补充实验于初次测试后进行，固定主模型与参数，作为解释和稳健性证据。

## 数据来源和许可

- 官方页面：https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality
- DOI：https://doi.org/10.24432/C5RK5G
- 下载：https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
- 引用：Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository.
- 数据许可：CC BY 4.0，https://creativecommons.org/licenses/by/4.0/ 。保留来源和署名；项目中的原始数据受其原许可约束。
- 下载日期、SHA256、原始归档层次记录于 `data/SOURCE.json`。

外层 ZIP 中还有网页附件和零散 CSV；程序仅读取其中官方 `PRSA2017_Data_20130301-20170228.zip` 的 12 个站点 CSV，避免误把其他文件拼接进去。原始归档保留且不修改。完整数据为 420,768 行、18 列，2013-03-01 至 2017-02-28。

## 目录

```text
data/raw/beijing-air-quality.zip   原始 UCI 归档（含12站点CSV）
data/processed/daily.csv           17,532条站点日聚合结果
data/processed/model_table.csv     特征、目标、日期和覆盖小时数
data/SOURCE.json                   来源、许可和文件哈希
src/data.py                       类、异常、数据审计和时间特征
src/modeling.py                    模型、调参、评价和块自助法
src/plots.py                       全部图表
run_analysis.py                    一键运行数据分析与实验
download_data.py                   可选：重新下载并校验数据
extended_analysis.py               重训敏感性、块长、事件与变化诊断
build_report.py                    根据分析结果生成PDF报告
analysis.ipynb                     可交互阅读的分析与结果展示
tests/test_project.py              时间泄漏、目标对齐和结果一致性检查
results/                          全部指标、预测、调参、误差表和图表
report/                           课程要求的PDF文件
requirements.txt                  固定核心依赖版本
requirements-notebook.txt         可选 Notebook 依赖
project_config.json               姓名、学号、GitHub链接
```

## 运行环境与复现

建议使用 Python 3.10。实际实验环境、操作系统、包版本、随机种子和耗时记录在 `results/environment.json`。随机种子统一为 42；随机森林用 4 个线程；其他数值计算限制为 4 个线程。不同平台或底层库可能产生微小浮点差异。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python run_analysis.py
python extended_analysis.py
python -m unittest discover -s tests -v
python build_report.py
```

macOS/Linux 激活环境用 `source .venv/bin/activate`，后续命令相同。已附原始数据时，下载命令只校验并复用数据，不需要联网。训练耗时依机器而异，通常为数分钟。

PDF 中文字体：脚本优先使用 Windows 的宋体/黑体；非 Windows 环境可以通过环境变量 `REPORT_FONT` 和 `REPORT_BOLD_FONT` 指定支持中文的 TrueType 字体文件。图表可通过系统安装 Noto Sans CJK SC / 微软雅黑 / 黑体显示中文。字体是本机系统资源，不打包或重新分发。

可选 Notebook：

```bash
python -m pip install -r requirements-notebook.txt
jupyter notebook analysis.ipynb
```

Notebook 默认展示随附结果；如需全部重算，将第一个运行代码单元中的 `RETRAIN` 设为 `True`。完整复现请按上述顺序运行命令行脚本。

打包完整项目（需安装上述可选 Notebook 依赖）：

```bash
python tools/package_project.py
```

脚本先运行测试；测试失败或未发现测试时停止打包。实际测试计数与文件检查结果写入 `results/verification.json`，不自动声明已完成人工排版检查。ZIP 输出至 `output/python_final_2026/`，仅包含指定项目文件、正式报告、数据和结果，不包含临时目录、修订版报告或旧压缩包。

## 数据处理原则

- 重复“站点-小时”键直接报错，不静默保留第一条；日期错误报错。
- 污染物、降水、风速负值转缺失并记录数量；负气温、负露点合法。
- 不把缺失值填 0，不跨日期向后填充，不使用全样本均值插补。
- 日均值至少需要 18 个有效小时；PM2.5 目标和当日 PM2.5 均不足时排除该预测样本。降雨使用有效小时降水量之和；18至23小时的部分日总量可能低估真实全天总量。
- 晚间 PM2.5 使用 18:00-23:00，有效小时至少为 4。日极值和标准差也要求至少18小时。
- 保留极端浓度，不根据箱线图删去污染事件；最高小时值不代表已排除仪器上限或异常记录的可能。
- 先补齐“站点-日”网格再 shift(-1)，保证目标严格是下一日，而不是下一个有记录的日期。
- 风向映射为 sin/cos 后取均值，避免把北和北北西算成相反方向。
- 中位数插补、缺失指示、标准化、类别编码都在 Pipeline 内，只用各训练折拟合。

## 预测和验证设计

主模型使用30个原始特征：11项PM2.5历史与当日统计、5项站点/日历特征、8项气象特征、5项其他污染物及1项区域PM2.5均值。缺失指示和独热编码会增加变换后的列数。

- 开发目标日期：2013-03-02 至 2016-02-28。
- 独立测试目标日期：2016-03-01 至 2017-02-28，完整365日。
- 2016-02-29的目标样本从最终训练中排除，保证训练标签早于首个测试样本的特征日期。
- 开发期内按日期扩展训练窗口，验证块为2015年3-6月、7-10月及2015年11月至2016年2月。每折也留1日间隔，同一天全部站点保持在同一划分。
- 以三折RMSE的**算术平均**选模型和参数，不使用测试集选择模型。
- 最终模型参数在测试年固定，随每天新观测构造下一日特征。这是逐日一步预测，不是在测试年开始时一次预测未来365天，也不进行逐日重新训练。
- HistGradientBoosting 显式关闭默认的随机内部早停验证；240轮固定，在时间CV中比较参数。
- 五种方法：训练均值、持续性、Ridge、RandomForest、HistGradientBoosting。
- 4+4+8=16组超参数候选，每组3折，共48次候选拟合，另含各最优模型全开发期拟合、基线和解释实验。
- 所有方法在完全相同的测试样本上比较。所有预测截断至非负，且该规则在CV中保持一致。

## 额外分析

- 三组特征消融：历史+日历、再加气象、全部特征。固定CV选定的算法及全特征最优参数，不重新调参，所以只能衡量这组固定参数下的特征增益。
- 最后一折验证集上做5次置换重要性；相关变量会分摊重要性，置换可能产生不自然组合，不作因果解释。
- 高污染分析阈值取开发目标的90%分位数，不是国家标准或AQI等级。
- 以7个连续日为块做1,500次配对自助抽样，同一天所有站点共同抽取，报告RMSE差的95%区间。它反映给定模型在该测试期的评估不确定性，不包含重新训练或未来年份漂移。
- 原模型在“当日和次日均至少20小时”子集复评；另将18/20/24小时门槛从小时数据重新聚合、重建特征并训练，比较各自合格样本和共同样本。
- 在固定开发期90%分位阈值上，将回归输出转换为经验高浓度事件提示，比较精确率、召回率、F1、误报与漏报。
- 按月份和次日真实变化幅度诊断误差。变化组由事后观测确定，用于解释模型适用条件。

## 结果文件说明

`metrics.csv` 为最终比较；`cv_*.csv` 保存每个参数候选的逐折训练/验证分数；`model_selection.json` 记录仅基于CV选定的模型；`test_predictions.csv` 为所有样本的实际值和五种预测；`subgroups.csv` 为分组误差；`bootstrap.json` 为不确定性；`ablation.csv` 与 `permutation_importance.csv` 为解释实验。各表由分析脚本生成。

## 适用范围

数据为历史时期的12站点，不代表北京市所有空间位置，也不代表当前空气质量。使用已匹配气象站的观测，不是明日天气预报。模型在当日结束后发布，假定观测可及时获得；真实部署应处理发布延迟、设备故障与概念漂移。气象关联采用观测性解释，政策效果需要额外的对照设计。

## 官方方法参考

- Pipeline：https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html
- Ridge：https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html
- RandomForestRegressor：https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- HistGradientBoostingRegressor：https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- GridSearchCV：https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GridSearchCV.html
- Permutation importance：https://scikit-learn.org/1.7/modules/permutation_importance.html
