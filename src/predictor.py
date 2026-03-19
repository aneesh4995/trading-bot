import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


FEATURES = ["RSI_14", "MACD", "MACD_signal", "EMA_20", "EMA_50", "EMA_200", "ATR_14"]


class SwingPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        self.trained = False

    def _make_target(self, df: pd.DataFrame) -> pd.Series:
        future_close = df["close"].shift(-5)
        return (future_close > df["close"] * 1.005).astype(int)

    def train(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["target"] = self._make_target(df)
        df = df.dropna(subset=FEATURES + ["target"])

        X = df[FEATURES]
        y = df["target"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        self.trained = True

    def predict(self, df: pd.DataFrame) -> tuple[int, float]:
        if not self.trained:
            raise RuntimeError("Model must be trained before predicting.")
        latest = df[FEATURES].dropna().iloc[[-1]]
        label = int(self.model.predict(latest)[0])
        prob = float(self.model.predict_proba(latest)[0][label])
        return label, prob
