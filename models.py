"""models.py - OOP layer: dataset generators (inheritance/polymorphism) and the ML model."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DatasetGenerator:
    """Base class. Child classes override generate()."""
    label = "approved"
    numeric_features = []
    sensitive_columns = ["gender", "region"]

    def __init__(self, n=3000, bias_strength=1.0, seed=42):
        self.n = n
        self.bias_strength = bias_strength
        self._rng = np.random.default_rng(seed)        # private member

    def generate(self):
        raise NotImplementedError("Child class must implement generate()")

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _build(self, gender, region, features, merit, offset, penalty):
        """Shared by children: turn merit + bias penalty into approval labels."""
        p = self._sigmoid(merit + offset + self.bias_strength * penalty)
        df = pd.DataFrame({"gender": gender, "region": region, **features,
                           self.label: (self._rng.random(self.n) < p).astype(int)})
        df.loc[self._rng.random(self.n) < 0.02, "income"] = np.nan   # missing values on purpose
        return df


class LoanDataGenerator(DatasetGenerator):
    numeric_features = ["credit_score", "income", "employment_years"]

    def generate(self):
        r, n = self._rng, self.n
        gender = r.choice(["Male", "Female"], n, p=[0.55, 0.45])
        region = r.choice(["Urban", "Rural"], n, p=[0.6, 0.4])
        credit = np.clip(r.normal(650, 70, n), 300, 850).round(0)
        # income is a PROXY: it differs by gender, so hiding 'gender' will not fully remove bias
        income = np.clip(r.normal(48000, 14000, n) + 6000 * (gender == "Male"), 10000, None).round(0)
        years = r.integers(0, 25, n)
        merit = 0.012 * (credit - 650) + 0.00004 * (income - 50000) + 0.03 * years
        penalty = -1.0 * (gender == "Female") - 0.6 * (region == "Rural")
        return self._build(gender, region, {"credit_score": credit, "income": income,
                                            "employment_years": years}, merit, 0.2, penalty)


class HiringDataGenerator(DatasetGenerator):
    numeric_features = ["test_score", "income", "employment_years"]

    def generate(self):
        r, n = self._rng, self.n
        gender = r.choice(["Male", "Female"], n, p=[0.6, 0.4])
        region = r.choice(["Urban", "Rural"], n, p=[0.5, 0.5])
        test = np.clip(r.normal(60, 15, n), 0, 100).round(1)
        income = np.clip(r.normal(40000, 12000, n) + 5000 * (gender == "Male"), 8000, None).round(0)  # expected salary
        years = r.integers(0, 20, n)
        merit = 0.05 * (test - 60) + 0.00002 * (income - 40000) + 0.05 * years
        penalty = -0.9 * (gender == "Female") - 0.5 * (region == "Rural")
        return self._build(gender, region, {"test_score": test, "income": income,
                                            "employment_years": years}, merit, -0.3, penalty)


GENERATORS = {"Loan approval": LoanDataGenerator, "Hiring": HiringDataGenerator}


class BiasModel:
    """Logistic regression with optional sensitive columns and sample weights."""

    def __init__(self, numeric_features, sensitive_cols=None):
        self.numeric = list(numeric_features)
        self.sensitive = list(sensitive_cols or [])
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self._columns = None

    def _X(self, df):
        X = df[self.numeric].copy()
        if self.sensitive:
            X = X.join(pd.get_dummies(df[self.sensitive], drop_first=True).astype(int))
        return X

    def fit(self, df, label, sample_weight=None):
        X = self._X(df)
        self._columns = list(X.columns)
        kw = {} if sample_weight is None else {"logisticregression__sample_weight": sample_weight}
        self.pipe.fit(X, df[label], **kw)
        return self

    def predict(self, df):
        return self.pipe.predict(self._X(df).reindex(columns=self._columns, fill_value=0))
