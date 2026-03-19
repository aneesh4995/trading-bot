import pandas as pd


class OptionsSignalMapper:
    CONVICTION_THRESHOLD = 0.65
    ATR_PERCENTILE = 60

    def get_strategy(
        self,
        prediction: int,
        probability: float,
        atr_series: pd.Series,
    ) -> dict:
        current_atr = float(atr_series.iloc[-1])
        atr_threshold = float(atr_series.rolling(20).quantile(self.ATR_PERCENTILE / 100).iloc[-1])
        high_volatility = current_atr > atr_threshold

        direction = "Up" if prediction == 1 else "Down"

        if probability < self.CONVICTION_THRESHOLD:
            return {
                "strategy": "No Trade / Wait",
                "direction": direction,
                "rationale": f"Low conviction ({probability:.1%}); wait for clearer signal.",
            }

        if direction == "Up" and not high_volatility:
            strategy, rationale = "Long Call", "High conviction bullish with low volatility — cheap premium."
        elif direction == "Up" and high_volatility:
            strategy, rationale = "Bull Call Spread", "High conviction bullish with high volatility — spread limits cost."
        elif direction == "Down" and high_volatility:
            strategy, rationale = "Bear Put Spread", "High conviction bearish with high volatility — spread limits cost."
        else:
            strategy, rationale = "Long Put", "High conviction bearish with low volatility — cheap premium."

        return {
            "strategy": strategy,
            "direction": direction,
            "rationale": rationale,
        }
