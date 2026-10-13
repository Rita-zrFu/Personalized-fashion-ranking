"""Behavioral baselines, small validation grids and validation-only calibration."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .customer_features import model_feature_columns


def signed_log_features(values):
    # Preserve negative momentum while reducing the long tails of count/gap/spend features.
    return np.sign(values) * np.log1p(np.abs(values))


def trial_settings(name):
    if name == "logistic_log":
        return [dict(C=.1), dict(C=1.)]
    if name in ["xgboost", "lightgbm"]:
        return [dict(depth=3), dict(depth=5)]
    return [dict()]


def heuristic_score(frame):
    return np.exp(-frame.recency_days.to_numpy() / 28) * (1 + np.log1p(frame.active_days.to_numpy()))


def fit_model(name, train, config, settings=None):
    import lightgbm as lgb
    import xgboost as xgb
    settings = settings or {}
    columns = model_feature_columns(train)
    x, y = train[columns], train.target
    prior = float(y.mean())
    common = dict(n_estimators=config["n_estimators"], random_state=config["seed"], n_jobs=config["n_jobs"])
    if name in ["population_prior", "recency_frequency"]:
        return dict(name=name, model=None, columns=columns, settings=settings, prior=prior,
                    heuristic_mean=float(heuristic_score(train).mean()))
    if name in ["logistic_raw", "logistic_log"]:
        steps = [StandardScaler(), LogisticRegression(C=settings.get("C", 1.), max_iter=1000)]
        if name == "logistic_log":
            steps.insert(0, FunctionTransformer(signed_log_features))
        model = make_pipeline(*steps)
    elif name == "random_forest":
        model = RandomForestClassifier(**common, max_depth=8, min_samples_leaf=30)
    elif name == "xgboost":
        model = xgb.XGBClassifier(**common, tree_method="hist", objective="binary:logistic", eval_metric="logloss",
                                 max_depth=settings.get("depth", 3), learning_rate=.05, reg_lambda=3.)
    elif name == "lightgbm":
        depth = settings.get("depth", 3)
        model = lgb.LGBMClassifier(**common, max_depth=depth, num_leaves=2**depth,
                                  objective="binary", learning_rate=.05, reg_lambda=3.,
                                  min_child_samples=40, verbosity=-1)
    else:
        raise ValueError(name)
    model.fit(x, y)
    return dict(name=name, model=model, columns=columns, settings=settings, prior=prior)


def predict_raw(fitted, frame):
    if fitted["name"] == "population_prior":
        return np.full(len(frame), fitted["prior"])
    if fitted["name"] == "recency_frequency":
        return np.clip(fitted["prior"] * heuristic_score(frame) / fitted["heuristic_mean"], 1e-6, 1-1e-6)
    return np.clip(fitted["model"].predict_proba(frame[fitted["columns"]])[:, 1], 1e-6, 1-1e-6)


def probability_logit(probabilities):
    values = np.clip(np.asarray(probabilities), 1e-6, 1-1e-6)
    return np.log(values / (1-values)).reshape(-1, 1)


def fit_calibrator(validation_probabilities, target):
    calibrator = LogisticRegression(C=1000., max_iter=500)
    calibrator.fit(probability_logit(validation_probabilities), target)
    return calibrator


def predict_calibrated(calibrator, raw_probabilities):
    return calibrator.predict_proba(probability_logit(raw_probabilities))[:, 1]
