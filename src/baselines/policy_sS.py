# src/baselines/policy_sS.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SsParams:
    s: int   # reorder point
    S: int   # order-up-to level
    expedite_threshold: float = 0.85  # optionally expedite when SCRI too high
    mitigate_disruption_level: int = 2  # optionally mitigate on severe disruption

class SsPolicy:
    """
    Classic (s, S) policy adapted to your env action dict:
      action = {"order_qty": q, "expedite": {0,1}, "mitigate": {0,1}}
    Refs: Your MDP spec (actions: order, expedite, mitigate) and KPI definitions. 
    """

    def __init__(self, params: SsParams):
        self.params = params

    def act(self, obs):
        """
        obs = [inventory, outstanding, leadtime, disruption, scri] (float32)
        See SupplyChainSimEnv._get_obs() in your env. 
        """
        inv, outstanding, leadtime, disruption, scri = obs
        inv = int(round(float(inv)))
        disruption = int(round(float(disruption)))
        scri = float(scri)

        q = 0
        if inv <= self.params.s:
            q = max(0, int(self.params.S - inv))

        expedite = 1 if scri >= self.params.expedite_threshold else 0
        mitigate = 1 if disruption >= self.params.mitigate_disruption_level else 0

        return {"order_qty": q, "expedite": expedite, "mitigate": mitigate}
