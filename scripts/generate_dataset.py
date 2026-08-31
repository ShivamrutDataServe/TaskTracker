"""
One-time script that generates data/churn.csv.

This is NOT part of the CI/CD pipeline - it's run once, by a human, and
the resulting CSV is committed to the repo (exactly like a real churn
dataset exported from a data warehouse would be). The pipeline itself
only ever reads data/churn.csv; it never regenerates it. That keeps
every CI run training on identical data, so metric changes always come
from code changes, not from random data drift.

Re-run this only if you deliberately want a new synthetic dataset:
    python scripts/generate_dataset.py
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N_ROWS = 2000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    tenure_months = RNG.integers(0, 73, N_ROWS)
    monthly_charges = np.round(RNG.uniform(20, 120, N_ROWS), 2)
    # total_charges roughly tracks tenure * monthly, with noise - like a
    # real billing system rather than a perfectly clean multiplication.
    total_charges = np.round(
        tenure_months * monthly_charges * RNG.uniform(0.85, 1.05, N_ROWS), 2
    )
    # 0 = month-to-month, 1 = one_year, 2 = two_year
    contract_type = RNG.choice([0, 1, 2], size=N_ROWS, p=[0.55, 0.25, 0.20])
    support_calls = RNG.poisson(1.5, N_ROWS).clip(0, 10)
    is_senior_citizen = RNG.choice([0, 1], size=N_ROWS, p=[0.84, 0.16])
    has_tech_support = RNG.choice([0, 1], size=N_ROWS, p=[0.5, 0.5])

    # A realistic-but-synthetic churn signal: longer tenure and longer
    # contracts reduce churn risk; more support calls, being a senior
    # citizen, and higher monthly charges increase it. Noise keeps classes
    # from being perfectly separable, like real churn data.
    logit = (
        -0.035 * tenure_months
        + 0.018 * monthly_charges
        + 0.28 * support_calls
        - 0.9 * contract_type
        + 0.55 * is_senior_citizen
        - 0.45 * has_tech_support
        + RNG.normal(0, 0.6, N_ROWS)
    )
    churn_probability = sigmoid(logit)
    churn = RNG.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "support_calls": support_calls,
            "is_senior_citizen": is_senior_citizen,
            "has_tech_support": has_tech_support,
            "churn": churn,
        }
    )
    df.to_csv("data/churn.csv", index=False)
    print(f"Wrote data/churn.csv with {len(df)} rows")
    print(f"Churn rate: {df['churn'].mean():.1%}")


if __name__ == "__main__":
    main()
