import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Original features (backward compatible)
FEATURES_V1 = ["RSI_14", "MACD", "MACD_signal", "EMA_20", "EMA_50", "EMA_200", "ATR_14"]

# V2 features: add VIX, put/call, volume ratio, EMA distance, day of week, RSI divergence
FEATURES_V2 = FEATURES_V1 + [
    "volume_ratio", "dist_ema200", "day_of_week", "rsi_divergence",
    "VIX", "put_call_ratio",
]

# Alias for backward compatibility
FEATURES = FEATURES_V1


class SwingPredictor:
    """Original single-model predictor. Kept for backward compatibility."""

    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        self.trained = False

    def _make_target(self, df: pd.DataFrame) -> pd.Series:
        future_close = df["close"].shift(-5)
        return (future_close > df["close"] * 1.005).astype(int)

    def train(self, df: pd.DataFrame) -> None:
        features = [f for f in FEATURES_V2 if f in df.columns] or FEATURES_V1
        df = df.copy()
        df["target"] = self._make_target(df)
        df = df.dropna(subset=features + ["target"])

        X = df[features]
        y = df["target"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        self._features = features
        self.trained = True

    def predict(self, df: pd.DataFrame, sentiment_score: float = None) -> tuple[int, float]:
        if not self.trained:
            raise RuntimeError("Model must be trained before predicting.")
        features = self._features
        latest = df[features].dropna().iloc[[-1]].copy()

        label = int(self.model.predict(latest)[0])
        prob = float(self.model.predict_proba(latest)[0][label])

        if sentiment_score is not None:
            if label == 1:
                adjustment = sentiment_score * 0.10
            else:
                adjustment = -sentiment_score * 0.10
            prob = max(0.01, min(0.99, prob + adjustment))

        return label, prob


class EnsemblePredictor:
    """
    Ensemble of 3 models with majority voting:
      1. GradientBoosting (strong on non-linear patterns)
      2. RandomForest (reduces overfitting via bagging)
      3. LogisticRegression (linear baseline, captures simple trends)

    Uses V2 features including VIX, put/call ratio, volume ratio.
    """

    def __init__(self):
        self.models = {
            "gb": GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
            "rf": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            "lr": LogisticRegression(max_iter=1000, solver="saga", random_state=42),
        }
        self.scaler = StandardScaler()  # needed for LogisticRegression
        self.trained = False
        self._features = []

    def _make_target(self, df: pd.DataFrame) -> pd.Series:
        future_close = df["close"].shift(-5)
        return (future_close > df["close"] * 1.005).astype(int)

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        return [f for f in FEATURES_V2 if f in df.columns] or FEATURES_V1

    def train(self, df: pd.DataFrame) -> None:
        self._features = self._select_features(df)
        df = df.copy()
        df["target"] = self._make_target(df)
        df = df.dropna(subset=self._features + ["target"])

        X = df[self._features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df["target"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Scale for LogisticRegression
        X_train_scaled = self.scaler.fit_transform(X_train)

        for name, model in self.models.items():
            if name == "lr":
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)

        self.trained = True

    def predict(self, df: pd.DataFrame, sentiment_score: float = None) -> tuple[int, float]:
        if not self.trained:
            raise RuntimeError("Ensemble must be trained before predicting.")

        latest = df[self._features].replace([np.inf, -np.inf], np.nan).fillna(0).iloc[[-1]].copy()
        latest_scaled = self.scaler.transform(latest)

        votes = []
        probs = []
        for name, model in self.models.items():
            if name == "lr":
                label = int(model.predict(latest_scaled)[0])
                prob = float(model.predict_proba(latest_scaled)[0][label])
            else:
                label = int(model.predict(latest)[0])
                prob = float(model.predict_proba(latest)[0][label])
            votes.append(label)
            probs.append(prob)

        # Majority vote
        final_label = 1 if sum(votes) >= 2 else 0

        # Average probability of the winning class across models that agree
        agreeing_probs = [p for v, p in zip(votes, probs) if v == final_label]
        final_prob = float(np.mean(agreeing_probs))

        # Sentiment adjustment
        if sentiment_score is not None:
            if final_label == 1:
                adjustment = sentiment_score * 0.10
            else:
                adjustment = -sentiment_score * 0.10
            final_prob = max(0.01, min(0.99, final_prob + adjustment))

        return final_label, final_prob

    def predict_detail(self, df: pd.DataFrame) -> dict:
        """Return per-model predictions for transparency."""
        latest = df[self._features].replace([np.inf, -np.inf], np.nan).fillna(0).iloc[[-1]].copy()
        latest_scaled = self.scaler.transform(latest)

        detail = {}
        for name, model in self.models.items():
            if name == "lr":
                label = int(model.predict(latest_scaled)[0])
                prob = float(model.predict_proba(latest_scaled)[0][label])
            else:
                label = int(model.predict(latest)[0])
                prob = float(model.predict_proba(latest)[0][label])
            detail[name] = {"label": label, "direction": "Up" if label == 1 else "Down", "prob": prob}
        return detail


class WalkForwardPredictor:
    """
    Walk-forward retraining: trains on expanding window, predicts next period.
    Retrains every `retrain_every` days to adapt to changing market regimes.

    Uses EnsemblePredictor internally.
    """

    def __init__(self, retrain_every: int = 63):  # ~quarterly
        self.retrain_every = retrain_every
        self.ensemble = EnsemblePredictor()

    def _make_target(self, df: pd.DataFrame) -> pd.Series:
        future_close = df["close"].shift(-5)
        return (future_close > df["close"] * 1.005).astype(int)

    def backtest(self, df: pd.DataFrame, min_train: int = 252) -> pd.DataFrame:
        """
        Walk-forward backtest. Returns DataFrame with columns:
        date, label, prob, actual, correct
        """
        features = [f for f in FEATURES_V2 if f in df.columns] or FEATURES_V1
        df = df.copy()
        df["target"] = self._make_target(df)
        df = df.dropna(subset=features + ["target"])

        results = []
        i = min_train

        while i < len(df) - 5:
            # Train on all data up to i
            train_df = df.iloc[:i]
            self.ensemble = EnsemblePredictor()
            self.ensemble.train(train_df)

            # Predict on day i
            pred_df = df.iloc[:i + 1]
            label, prob = self.ensemble.predict(pred_df)
            actual = int(df.iloc[i]["target"])

            results.append({
                "date": df.index[i],
                "label": label,
                "prob": prob,
                "actual": actual,
                "correct": label == actual,
            })

            # Step forward
            i += self.retrain_every

        return pd.DataFrame(results)

    def train_latest(self, df: pd.DataFrame) -> None:
        """Train ensemble on all available data for live prediction."""
        self.ensemble = EnsemblePredictor()
        self.ensemble.train(df)

    def predict(self, df: pd.DataFrame, sentiment_score: float = None) -> tuple[int, float]:
        return self.ensemble.predict(df, sentiment_score)

    def predict_detail(self, df: pd.DataFrame) -> dict:
        return self.ensemble.predict_detail(df)
