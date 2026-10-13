"""Pointwise classifiers and customer-week grouped learning-to-rank models."""
from __future__ import annotations

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_GROUPS

ALL_FEATURES = list(dict.fromkeys(c for g in FEATURE_GROUPS.values() for c in g))


def fit_model(name, frame, config, depth=5, features=None):
    import lightgbm as lgb
    import xgboost as xgb
    features = list(features or ALL_FEATURES)
    frame = frame.sort_values(["query_id", "article_id"]).reset_index(drop=True)
    x, y = frame[features], frame.label
    common = dict(n_estimators=config["n_estimators"], random_state=config["seed"], n_jobs=4)
    start = time.perf_counter()
    if name == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=.1))
        model.fit(x, y)
    elif name == "random_forest":
        model = RandomForestClassifier(**common, max_depth=depth + 3, min_samples_leaf=20)
        model.fit(x, y)
    elif name in ["lightgbm_classifier", "lambdamart"]:
        params = dict(**common, max_depth=depth, num_leaves=2**depth, learning_rate=.05,
                      min_child_samples=20, reg_lambda=1., verbosity=-1)
        if name == "lambdamart":
            model = lgb.LGBMRanker(**params, objective="lambdarank", lambdarank_truncation_level=20)
            group = frame.groupby("query_id", sort=False).size().to_numpy()
            model.fit(x, y, group=group)
        else:
            model = lgb.LGBMClassifier(**params, objective="binary")
            model.fit(x, y)
    elif name in ["xgboost_classifier", "xgb_pairwise", "xgb_ndcg"]:
        params = dict(**common, max_depth=depth, learning_rate=.05, reg_lambda=1.,
                      tree_method="hist", subsample=1., colsample_bytree=1.)
        if name == "xgboost_classifier":
            model = xgb.XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss")
            model.fit(x, y)
        else:
            objective = "rank:pairwise" if name == "xgb_pairwise" else "rank:ndcg"
            model = xgb.XGBRanker(**params, objective=objective)
            qid = frame.groupby("query_id", sort=False).ngroup().to_numpy()
            model.fit(x, y, qid=qid)
    else:
        raise ValueError(name)
    return dict(name=name, model=model, features=features, depth=depth,
                train_seconds=time.perf_counter() - start)


def predict(fitted, frame):
    x = frame[fitted["features"]]
    if fitted["name"] in ["lambdamart", "xgb_pairwise", "xgb_ndcg"]:
        return fitted["model"].predict(x)
    return fitted["model"].predict_proba(x)[:, 1]


def importance(fitted):
    import pandas as pd
    model = fitted["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif fitted["name"] == "logistic":
        values = np.abs(model[-1].coef_[0])
    else:
        return pd.DataFrame()
    return pd.DataFrame({"feature": fitted["features"], "importance": values}).sort_values("importance", ascending=False)
