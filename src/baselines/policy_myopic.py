# src/baselines/policy_myopic.py
from dataclasses import dataclass

@dataclass(frozen=True)
class MyopicParams:
    safety_factor: float = 0.0        # optional buffer over forecast
    expedite_threshold: float = 0.9   # avoid big one-step penalties if SCRI gets high
    mitigate_on_disruption: int = 2   # mitigate only on severe disruption

class MyopicPolicy:
    """
    Myopic one-step look: order to cover current forecast (approx) using state proxy.
    Uses demand_forecast as provided by the env each step (via info pattern).
    Since env.step returns 'info', but policy sees only obs, we approximate using:
      - recent obs inventory vs. 'implied demand' from stockout/scri risk proxy.
    Practical variant: order to match 'target' = max(current inv, 0) + small buffer.
    """

    def __init__(self, params: MyopicParams):
        self.params = params
        self._last_forecast = None

    def set_forecast(self, demand_forecast: int):
        self._last_forecast = int(demand_forecast)

    def act(self, obs):
        inv, outstanding, leadtime, disruption, scri = obs
        inv = int(round(float(inv)))
        disruption = int(round(float(disruption)))
        scri = float(scri)

        # Heuristic demand proxy when forecast not directly in obs:
        forecast = self._last_forecast if self._last_forecast is not None else 10

        target = int((1.0 + self.params.safety_factor) * forecast)
        q = max(0, target - inv)

        expedite = 1 if scri >= self.params.expedite_threshold else 0
        mitigate = 1 if disruption >= self.params.mitigate_on_disruption else 0
        return {"order_qty": q, "expedite": expedite, "mitigate": mitigate}
