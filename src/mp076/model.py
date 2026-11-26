"""Models accept the same eight named inputs as the prediction service."""

from functools import partial
from math import ceil

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, DATE, FEATURES, NUMERIC_FEATURES, TARGET


def build_model(name, settings, random_state):
    regressors = {
        "ridge": Ridge,
        "huber": HuberRegressor,
        "gradient_boosting": HistGradientBoostingRegressor,
    }
    if name == "gradient_boosting":
        settings = {**settings, "random_state": random_state}
    numeric = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    # Cash withdrawals cannot be negative, including when a linear model extrapolates.
    regressor = TransformedTargetRegressor(
        regressor=regressors[name](**settings),
        func=np.copy,
        inverse_func=partial(np.maximum, 0),
        check_inverse=False,
    )
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def split_validation(frame, fraction):
    if len(frame) < 2 or not 0 < fraction < 1:
        raise ValueError("Chronological validation requires at least two rows and a fraction between 0 and 1.")
    frame = frame.sort_values(DATE, kind="stable")
    cutoff = frame[DATE].iloc[len(frame) - ceil(len(frame) * fraction)]
    is_validation = frame[DATE].ge(cutoff)
    if is_validation.all():
        raise ValueError("The validation cutoff must leave some earlier training dates.")
    return frame.loc[~is_validation], frame.loc[is_validation]


def evaluate_candidates(frame, training):
    fit, check = split_validation(frame, training["validation_fraction"])
    scores = {}
    for name, settings in training["candidates"].items():
        model = build_model(name, settings, training["random_state"])
        model.fit(fit[FEATURES], fit[TARGET])
        scores[name] = float(mean_absolute_error(check[TARGET], model.predict(check[FEATURES])))
    return scores, fit, check
