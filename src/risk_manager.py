ATR_MULTIPLIERS = {
    "conservative": {"stop": 1.5, "take_profit": 2.0},
    "moderate":     {"stop": 2.0, "take_profit": 3.0},
    "aggressive":   {"stop": 2.5, "take_profit": 4.0},
}

RECOMMENDED_DELTA = {
    "conservative": 0.70,
    "moderate":     0.55,
    "aggressive":   0.40,
}


class RiskManager:
    def __init__(self, account_size: float, risk_tolerance: str = "moderate"):
        if risk_tolerance not in ATR_MULTIPLIERS:
            raise ValueError(f"risk_tolerance must be one of {list(ATR_MULTIPLIERS)}")
        self.account_size = account_size
        self.risk_tolerance = risk_tolerance
        self._mults = ATR_MULTIPLIERS[risk_tolerance]

    def max_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_win == 0:
            return 0.0
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        half_kelly = max(kelly / 2, 0.0)
        return round(half_kelly * self.account_size, 2)

    def stop_loss(self, entry: float, atr: float) -> float:
        return round(entry - self._mults["stop"] * atr, 4)

    def take_profit(self, entry: float, atr: float) -> float:
        return round(entry + self._mults["take_profit"] * atr, 4)

    def recommended_delta(self) -> float:
        return RECOMMENDED_DELTA[self.risk_tolerance]
