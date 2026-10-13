# 本轮结果怎么解释

所有数字来自 2026-10-13 完成的真实 H&M 固定 50,000-user sample 实验。原始交易扫描 31,788,324 行；实际用于实验的是 1,156,381 条交易。最终测试周 2020-09-16 至 2020-09-22，有 2,452 位购买用户。

## 模型确实提高了排序，但不能把它写成 online business lift

Validation 选中的 XGBoost classifier 在 test 达到 MAP@12=0.0260748、Recall@12=0.0553021、NDCG@12=0.0386167。Recent-purchase baseline 的 MAP=0.0222364，relative improvement 为 17.3%；global decayed popularity 为 0.0050513。

与 recent baseline 的 paired-customer bootstrap：MAP 差为 0.0038384，95% interval 约 [0.0012181, 0.0064077]。这个区间只针对固定模型、样本和这一周的用户变异，不能证明其他周也稳定提升，更不能转换为转化率或销售提升。

LambdaMART test MAP=0.0249513，XGBoost NDCG=0.0244657，pairwise=0.0243169。排序 loss 并不自动优于 pointwise loss。当前 label 稀疏，训练中许多 query 没有被召回的 positive，有效排名样本有限。

两个 expanding validation folds 的赢家不同：早一周固定 depth 5 的 LambdaMART 更好，后一周 XGBoost classifier 更好。单周 validation 中 XGBoost 和 RF 的 MAP 只差约 0.00001，因此应将最终选择称为“本轮 validation-selected”，不要宣传算法普遍优越。

## 候选召回是最明显的瓶颈

40 popularity candidates 的 macro recall=5.34%，40 ALS candidates=4.71%，union=12.50%。每位用户平均约 147 个候选。Union 比单一来源好，但仍然漏掉大部分未来购买；改善 ranker 无法恢复候选池里不存在的商品。

在 7,376 个 distinct test customer–article targets 中，11.71% 的 article 在该样本 cutoff 前从未出现；13.08% 不在 120-day eligibility catalog 中。这部分也保留在评估 denominator，不能用未来购买记录反向补入候选。全量 population popularity/co-occurrence 或更好的 catalog eligibility 是后续值得研究的方向。

过去买过的商品只占 future distinct targets 的约 3.58%，但 recent baseline 仍然强：少数 repeat targets 往往容易排到列表前部，novel targets 则分散在大量商品中。候选中的历史购买次数和时间间隔因此仍有价值。

## Feature ablation 的结论

| 移除特征家族 | Test MAP@12 | 相对完整模型的绝对变化 |
|---|---:|---:|
| 无移除 | 0.02607 | — |
| ALS | 0.02641 | +0.00034 |
| Recency | 0.02418 | -0.00190 |
| Category/color/department affinity | 0.02522 | -0.00085 |
| Co-occurrence | 0.02485 | -0.00122 |
| Content | 0.02462 | -0.00146 |

Recency 在这些消融中影响最大。ALS 对扩大候选有贡献，但 ALS 分数未必改善这个 scorer 的排序。这里的 feature ablation 仍然保留检索后的候选 membership，不能解释为整条 ALS/content pipeline 的因果贡献。

## 对谁有效，付出什么代价

Zero-history test users 有 220 位，selected scorer MAP=0.00736；购买超过 10 次的用户有 1,896 位，MAP=0.02726。低活跃组样本较小，不能根据 76 位 1–2-purchase users 的较高 MAP 就推出低活跃用户普遍更适合该模型。

真正 cutoff 前未见商品的条件指标为 0，这是当前 eligibility rule 下不可检索的结果，不能声称 content 已解决 true item cold start。最近一周 first-observed 商品的 MAP=0.02421；first observation 是样本内首次出现，不是确认的上市时间。

Selected scorer 的 eligible-catalog coverage=14.06%，高于 global popularity 的 0.052%，但低于 recent baseline 的 34.87%。MAP 改善与 coverage 下降可以同时发生。不能只展示相对最弱 baseline 的 diversity 优势；需要明确与强 baseline 的 tradeoff。

## 下一次实验该做什么

先改善 candidate coverage，再扩大 customer cohort 或使用 population-level retrieval。用新的开发周比较 source quotas、history windows、negative ratios 和 CF/content 配置；本轮 final test 已经用于报告和诊断，不能再一边看它一边优化并仍称为 untouched test。

Online A/B proposal 已写好，但尚未部署或实验。没有 impression、CTR、库存和完整商业指标，当前结果只支持 offline relevance 的结论。
