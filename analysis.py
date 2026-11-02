"""analysis.py - NumPy / Pandas: cleaning, fairness metrics, mitigation, audit pipeline."""
import numpy as np
from sklearn.model_selection import train_test_split
from models import BiasModel


def clean(df):
    """Fill numeric missing values with the column median. Returns (clean_df, n_filled)."""
    df = df.copy()
    n_filled = int(df.isnull().sum().sum())
    for col in df.select_dtypes("number"):
        df[col] = df[col].fillna(df[col].median())
    return df, n_filled


def selection_rates(df, group, pred="pred"):
    return df.groupby(group)[pred].mean()


def disparate_impact(rates):
    """min/max selection rate. Below 0.80 fails the '80% rule'."""
    return float(rates.min() / rates.max()) if rates.max() > 0 else 1.0


def parity_difference(rates):
    return float(rates.max() - rates.min())


def equal_opportunity_difference(df, group, label="approved", pred="pred"):
    """Gap in true-positive rate: are equally qualified people treated equally?"""
    tpr = df[df[label] == 1].groupby(group)[pred].mean()
    return float(tpr.max() - tpr.min())


def reweighing_weights(df, group, label="approved"):
    """Kamiran & Calders reweighing: w = P(group) * P(label) / P(group, label)."""
    p_g = df[group].map(df[group].value_counts(normalize=True))
    p_y = df[label].map(df[label].value_counts(normalize=True))
    p_gy = df.groupby([group, label])[label].transform("size") / len(df)
    return (p_g * p_y / p_gy).to_numpy()


def summarize(test_df, group, label="approved"):
    rates = selection_rates(test_df, group)
    return {"rates": rates,
            "accuracy": float((test_df[label] == test_df["pred"]).mean()),
            "di": disparate_impact(rates),
            "spd": parity_difference(rates),
            "eod": equal_opportunity_difference(test_df, group, label)}


def run_audit(df, numeric_features, group, sensitive_cols=("gender", "region"),
              use_sensitive=True, mitigate=False, label="approved", seed=1):
    """Train on 70%, audit on the held-out 30%. Returns (summary, test_df)."""
    train, test = train_test_split(df, test_size=0.3, random_state=seed, stratify=df[label])
    cols = [c for c in sensitive_cols if c in df.columns] if use_sensitive else []
    model = BiasModel(numeric_features, cols)
    w = reweighing_weights(train, group, label) if mitigate else None
    model.fit(train, label, sample_weight=w)
    test = test.copy()
    test["pred"] = model.predict(test)
    return summarize(test, group, label), test


def verdict(di):
    return "PASS (>= 0.80)" if di >= 0.8 else "FAIL (< 0.80, biased)"
