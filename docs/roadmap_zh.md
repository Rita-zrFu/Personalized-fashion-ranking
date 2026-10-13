# Ranking 项目执行记录

项目位置：OneDrive-Emory / Internship / ranking。
更新：2026-10-13。已完成经典 pipeline 的第一轮真实数据实验；不是模拟数据。

## 本轮数据和结论

扫描完整 31,788,324 行原始交易；按固定 seed 的 customer ID hash 选择 50,000 位用户，保留他们全部 1,156,381 条历史交易。模型的 popularity/ALS/co-occurrence 也基于该样本。商品 metadata 保留全部 105,542 行。正式测试是 2020-09-16 至 2020-09-22，共 2,452 位有购买的用户。

验证集选择 XGBoost classifier 排序，测试 MAP@12=0.02607；recent-purchase baseline=0.02224；decayed-popularity baseline=0.00505。最终选择依据是 validation，不是 test。验证集 XGBoost 和 random forest 的 MAP 几乎持平，不应解读成 XGBoost 总是最好。

## 35 步状态

以下“完成”表示本轮 50,000-user development cohort 上已经实现和运行，不表示百万用户全量实验或生产部署。

| 步骤 | 内容 | 本轮状态 / 产物 |
|---:|---|---|
| 1 | 下载并读入三张表 | 完成；data/processed/hm_sample/manifest.json |
| 2 | EDA | 完成；reports/hm_sample/eda/ |
| 3 | Temporal split | 完成；两训练周、一验证周、一测试周；附两个 expanding validation folds |
| 4 | Ranking metrics | 完成；MAP/Recall/Precision/NDCG/MRR@12；手算例子核对 |
| 5 | Global popularity | 完成；28 天、14 天尺度的指数衰减版本 |
| 6 | Segment popularity | 完成；age 分组，unknown fallback；不推断 gender |
| 7 | Recent purchases | 完成；30 天尺度的 decay + popularity fallback |
| 8 | Item co-occurrence | 完成；binary shared-user cosine + 每商品 50 neighbors |
| 9 | Implicit ALS | 完成；sparse confidence-weighted ALS，24 factors、8 iterations |
| 10 | Content retrieval | 完成；TF-IDF + 24-dimensional SVD + cosine |
| 11 | Candidate union | 完成；每源最多 40、合集最多 240，保留 source flags/ranks |
| 12 | Candidate recall | 完成；各来源和 union；test union=12.50% |
| 13 | User features | 完成；age、RFM proxy、价格、channel、diversity |
| 14 | Item features | 完成；7/14/28-day sales、trend、age、price |
| 15 | User-item features | 完成；previous purchase、recency、category/color/department affinity |
| 16 | Source features | 完成；scores、flags、ranks、RRF、source count |
| 17 | Negative sampling | 完成；只从候选中抽 hard negatives，最多 1:10；eval 不抽样 |
| 18 | Logistic | 完成；标准化 logistic scorer |
| 19 | Random forest | 完成 |
| 20 | LightGBM classifier | 完成 |
| 21 | XGBoost classifier | 完成；本轮 validation winner |
| 22 | XGBRanker | 完成；rank:pairwise 及 rank:ndcg |
| 23 | LambdaMART | 完成；LightGBM lambdarank；与 XGBoost NDCG 版本分开比较 |
| 24 | Hyperparameter tuning | 完成初始小网格；LambdaMART depth 3/5；其余固定参数 |
| 25 | Final temporal test | 完成；metrics.csv，最终模型与本地推荐列表 |
| 26 | Candidate ablation | 完成；6 sources；固定 ranker，移除该源独有候选 |
| 27 | Feature ablation | 完成；5 families；重训 selected ranker |
| 28 | Cold start | 完成；zero-history、1–2 次、first-seen items、unseen future items |
| 29 | Customer segments | 完成；activity、age、category diversity |
| 30 | Diversity/coverage | 完成；catalog coverage、novelty、content diversity |
| 31 | Business interpretation | 完成；报告讨论 recall 瓶颈、recency、实际数据限制 |
| 32 | Online A/B proposal | 设计已完成；没有运行 online A/B test |
| 33 | Source code | 完成；模块化的 H&M 专用 Python 实现 |
| 34 | README / GitHub | README 和分步提交指南完成；仓库已 init，尚未 commit/push |
| 35 | Optional Two-Tower | 尚未实现；按原方案作为可选第二阶段 |

## 你从哪里开始看

1. README 看整体流程、数据范围和结果。
2. notebooks/01_eda.ipynb 理解数据和时间结构。
3. candidates.py 看每个 source，结合 candidate_recall.csv 判断哪个 source 有贡献。
4. features.py → models.py → pipeline.py 看 features、label、分组训练和模型选择。
5. notebooks/03_model_analysis.ipynb 看 segments 和 ablations；写清“为什么”和“对谁有效”。

## 之后真正值得继续做的工作

- 在新的开发时间窗调 retrieval quota、lookback、ALS confidence/factors；不要反复以本轮 test 调参。
- 把 population popularity/co-occurrence 与 sampled-user personalized features 分开，或扩大固定用户 cohort。这有机会改善长尾和新品覆盖。
- 对更多历史周做 temporal validation，检查 XGBoost/RF 的微小 validation 差距能否稳定复现。
- 比较 popularity 1/2/4-week 版本、negative ratio 1:5/1:10，以及 content metadata/description 的相对贡献。
- 只有真实数据与结果支持时才升级简历量化结论。可选 Two-Tower 在经典 pipeline 之后做。

不额外搭通用框架或移植到别的数据集。每一步围绕这份 H&M 数据和可复跑的实证结果展开。
