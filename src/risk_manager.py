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

# VIX regime → base allocation fraction
VIX_REGIMES = {
    "calm":     {"vix_max": 18, "alloc": 0.25},
    "normal":   {"vix_max": 25, "alloc": 0.20},
    "elevated": {"vix_max": 30, "alloc": 0.10},
    "extreme":  {"vix_max": 999, "alloc": 0.05},
}

# Consecutive losses before cutting size
LOSS_STREAK_THRESHOLD = 2
LOSS_STREAK_MULTIPLIER = 0.50


class AdaptiveSizer:
    """Position sizing that adapts to VIX regime and recent win/loss streak."""

    def __init__(self, account_size: float):
        self.account_size = account_size
        self.consecutive_losses = 0

    def record_result(self, won: bool) -> None:
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def _vix_alloc(self, vix: float) -> float:
        for regime in VIX_REGIMES.values():
            if vix <= regime["vix_max"]:
                return regime["alloc"]
        return 0.05

    def _vix_regime_name(self, vix: float) -> str:
        for name, regime in VIX_REGIMES.items():
            if vix <= regime["vix_max"]:
                return name
        return "extreme"

    def position_size(self, vix: float) -> float:
        alloc = self._vix_alloc(vix)

        # Cut size after consecutive losses
        if self.consecutive_losses >= LOSS_STREAK_THRESHOLD:
            alloc *= LOSS_STREAK_MULTIPLIER

        return round(alloc * self.account_size, 2)

    def sizing_detail(self, vix: float) -> dict:
        regime = self._vix_regime_name(vix)
        base_alloc = self._vix_alloc(vix)
        effective_alloc = base_alloc
        streak_cut = False

        if self.consecutive_losses >= LOSS_STREAK_THRESHOLD:
            effective_alloc *= LOSS_STREAK_MULTIPLIER
            streak_cut = True

        return {
            "vix": vix,
            "regime": regime,
            "base_alloc": base_alloc,
            "effective_alloc": effective_alloc,
            "streak_cut": streak_cut,
            "consecutive_losses": self.consecutive_losses,
            "position_size": round(effective_alloc * self.account_size, 2),
            "account_size": self.account_size,
        }


class RiskManager:
    def __init__(self, account_size: float, risk_tolerance: str = "moderate"):
        if risk_tolerance not in ATR_MULTIPLIERS:
            raise ValueError(f"risk_tolerance must be one of {list(ATR_MULTIPLIERS)}")
        self.account_size = account_size
        self.risk_tolerance = risk_tolerance
        self._mults = ATR_MULTIPLIERS[risk_tolerance]
        self.sizer = AdaptiveSizer(account_size)

    def max_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_win == 0:
            return 0.0
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        half_kelly = max(kelly / 2, 0.0)
        return round(half_kelly * self.account_size, 2)

    def adaptive_position_size(self, vix: float) -> float:
        return self.sizer.position_size(vix)

    def record_trade(self, won: bool) -> None:
        self.sizer.record_result(won)

    def stop_loss(self, entry: float, atr: float) -> float:
        return round(entry - self._mults["stop"] * atr, 4)

    def take_profit(self, entry: float, atr: float) -> float:
        return round(entry + self._mults["take_profit"] * atr, 4)

    def recommended_delta(self) -> float:
        return RECOMMENDED_DELTA[self.risk_tolerance]
