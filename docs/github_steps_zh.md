# 分阶段提交 GitHub

本地已经 `git init -b main`。目前没有 commit、没有 remote、没有 push。下面把已有工作整理为 5 个有实际内容的阶段。每阶段阅读对应代码、查看结果，准备好后执行该阶段的 commit 和 push。

在 `ranking` 文件夹执行。每次 commit 前用 `git diff --cached --stat` 看具体内容。数据、模型、虚拟环境和含 customer ID 的本地输出已经忽略。

## 先建立远端，只做一次

在 GitHub 建立空仓库，建议名 `personalized-fashion-ranking`。创建时保持仓库为空，README 等文件由本地提交。将下面 URL 替换为实际仓库地址。

```bash
cd /path/to/Personalized-fashion-ranking
git remote add origin https://github.com/YOUR_USERNAME/personalized-fashion-ranking.git
```

每次只运行一个阶段。`git commit` 保存该阶段的本地记录，随后的 `git push` 将这一阶段送到 GitHub。尚未 `git add` 的其他文件仍留在本地。

## Commit 1：数据、时间切分和实验定义

```bash
git add .gitignore pyproject.toml requirements.txt requirements-lock.txt Makefile configs data/README.md data/raw/.gitkeep data/processed/.gitkeep
git add src/ranking/__init__.py src/ranking/data.py src/ranking/metrics.py src/ranking/cli.py
git add docs/original_proposal_zh.md docs/experiment_design.md
git --no-pager diff --cached --stat
git commit -m "Set up H&M data preparation and temporal evaluation"
git push -u origin main
```

验证证据：真实数据 manifest；50,000 用户、1,156,381 交易；时间窗口和手算指标核对。

## Commit 2：候选生成和特征

```bash
git add src/ranking/candidates.py src/ranking/features.py
git --no-pager diff --cached --stat
git commit -m "Add hybrid H&M retrieval and historical ranking features"
git push
```

包括 six sources、真正 implicit ALS、TF-IDF/SVD、source provenance、candidate recall、hard-negative sampling。

## Commit 3：模型训练和完整实验

```bash
git add src/ranking/models.py src/ranking/pipeline.py src/ranking/cli.py
git add src/ranking/analysis.py src/ranking/rolling.py src/ranking/reporting.py
git --no-pager diff --cached --stat
git commit -m "Train pointwise and learning-to-rank models on temporal snapshots"
git push
```

此时 `make run`、`make rolling`、`make publish` 可从已准备的数据复跑整个实验。

## Commit 4：真实评估结果和解释

```bash
git add reports/hm_sample notebooks docs/online_ab_test.md docs/findings_zh.md
git --no-pager diff --cached --stat
git commit -m "Report held-out H&M ranking results and ablation analysis"
git push
```

只提交 aggregate artifacts，不提交 reports/runs 中的 customer predictions 或 candidate-level training matrices。

## Commit 5：项目主页和简历呈现

```bash
git add README.md docs/roadmap_zh.md docs/github_steps_zh.md docs/resume_bullets.md
git --no-pager diff --cached --stat
git commit -m "Document reproducible workflow and verified resume results"
git push
```

## 每阶段完成后

在 GitHub 的 commits 页面查看该阶段的文件变化。完成当前阶段后，再准备下一阶段。后续新增的调参、召回改进或 Two-Tower 实验也按真实改动单独 commit/push，并在描述中记录数据范围、方法和结果。
