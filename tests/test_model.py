"""
Model regression tests. These check properties of the *trained artifact*
itself, separate from the API that serves it - catching cases where the
API tests would still pass (model loads, returns a probability) but the
model itself has quietly gotten worse or behaves nonsensically.
"""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("metrics.json") as f:
    metrics = json.load(f)

with open("metrics/baseline_metrics.json") as f:
    baseline = json.load(f)

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "support_calls",
    "is_senior_citizen",
    "has_tech_support",
]


def _features(tenure_months, monthly_charges, total_charges,
              contract_type, support_calls, is_senior_citizen, has_tech_support):
    return pd.DataFrame(
        [[tenure_months, monthly_charges, total_charges, contract_type,
          support_calls, is_senior_citizen, has_tech_support]],
        columns=FEATURE_COLUMNS,
    )


def test_metrics_meet_baseline():
    """Belt-and-suspenders: the CI quality gate already enforces this, but
    keeping it as a test too means `pytest` alone always tells the truth
    about whether the current model.pkl is deployable."""
    assert metrics["f1_score"] >= baseline["min_f1_score"]
    assert metrics["roc_auc"] >= baseline["min_roc_auc"]


def test_probabilities_are_valid():
    features = _features(12, 70.0, 840.0, 0, 2, 0, 0)
    proba = model.predict_proba(features)[0]
    assert len(proba) == 2
    assert np.isclose(proba.sum(), 1.0)
    assert 0.0 <= proba[1] <= 1.0


def test_more_support_calls_increases_churn_risk():
    """All else equal, a customer who has called support 8 times should
    never look safer than one who has called 0 times - a flipped
    relationship here usually means a feature-order bug."""
    low_calls = model.predict_proba(_features(12, 70.0, 840.0, 0, 0, 0, 0))[0][1]
    high_calls = model.predict_proba(_features(12, 70.0, 840.0, 0, 8, 0, 0))[0][1]
    assert high_calls > low_calls


def test_longer_contract_reduces_churn_risk():
    """A two-year contract (2) should predict lower churn risk than
    month-to-month (0), holding everything else constant."""
    month_to_month = model.predict_proba(_features(12, 70.0, 840.0, 0, 2, 0, 0))[0][1]
    two_year = model.predict_proba(_features(12, 70.0, 840.0, 2, 2, 0, 0))[0][1]
    assert two_year < month_to_month


def test_tech_support_reduces_churn_risk():
    without_support = model.predict_proba(_features(12, 70.0, 840.0, 0, 2, 0, 0))[0][1]
    with_support = model.predict_proba(_features(12, 70.0, 840.0, 0, 2, 0, 1))[0][1]
    assert with_support < without_support


@pytest.mark.parametrize(
    "tenure,monthly,total,contract,calls,senior,support",
    [
        (0, 20.0, 0.0, 0, 0, 0, 0),
        (72, 120.0, 8640.0, 2, 10, 1, 1),
        (12, 70.0, 840.0, 1, 3, 0, 0),
    ],
)
def test_predict_never_returns_nan_or_out_of_range(
    tenure, monthly, total, contract, calls, senior, support
):
    proba = model.predict_proba(
        _features(tenure, monthly, total, contract, calls, senior, support)
    )[0][1]
    assert np.isfinite(proba)
    assert 0.0 <= proba <= 1.0
