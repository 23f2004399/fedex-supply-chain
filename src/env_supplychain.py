import gym
import numpy as np
import simpy
from gym import spaces
from scipy.stats import t, norm

class StudentTCopulaSampler:

    def __init__(self, marginals, corr, df=4, seed=None):
        self.marginals = marginals
        self.corr = corr
        self.df = df
        self.dim = len(marginals)
        self.rng = np.random.default_rng(seed)

    def sample(self, n=1):
        g = self.rng.standard_normal((n, self.dim))
        L = np.linalg.cholesky(self.corr)
        z = g @ L.T
        chi2 = self.rng.chisquare(self.df, n)[:, None]
        t_samples = z / np.sqrt(chi2 / self.df)

        u = t.cdf(t_samples, df=self.df)

        samples = np.column_stack([m.ppf(u[:, i]) for i, m in enumerate(self.marginals)])
        return samples

class SupplyChainSimEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None, seed=None):
        super().__init__()
        self.config = config or {}
        self.seed(seed)

        self.action_space = spaces.Dict({
            "order_qty": spaces.Discrete(21), 
            "expedite": spaces.Discrete(2),    
            "mitigate": spaces.Discrete(2),    
        })

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0.0]),
            high=np.array([100, 100, 30, 2, 1.0]),
            dtype=np.float32
        )

        self.marginals = [
            norm(loc=5, scale=2),      
            norm(loc=100, scale=30),   
            norm(loc=2, scale=0.5),   
        ]
        self.corr = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.4],
            [0.2, 0.4, 1.0]
        ])
        self.copula = StudentTCopulaSampler(self.marginals, self.corr, df=4, seed=self._seed)
        self.max_steps = self.config.get("max_steps", 30)
        self.reset()

    def seed(self, seed=None):
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self.env = simpy.Environment()
        self.current_step = 0
        self.inventory = 50
        self.outstanding = 0
        self.leadtime = 5
        self.disruption = 0
        self.scri = 0.0
        self.done = False
        self.total_cost = 0.0
        self.order_pipeline = []
        self.demand_forecast = 10
        self._setup_events()
        return self._get_obs()

    def _setup_events(self):
        self.next_demand_time = 0
        self.next_disruption_time = 10
        self.env.process(self._demand_process())
        self.env.process(self._disruption_process())

    def _demand_process(self):
        while True:
            _, severity, interarrival = self.copula.sample(1)[0]
            yield self.env.timeout(max(0.1, interarrival))
            self.demand_forecast = max(1, int(severity))

    def _disruption_process(self):
        while True:
            yield self.env.timeout(10 + self.np_random.integers(-2, 3))
            self.disruption = self.np_random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        qty = int(action["order_qty"])
        expedite = bool(action["expedite"])
        mitigate = bool(action["mitigate"])

        leadtime, _, _ = self.copula.sample(1)[0]
        leadtime = max(1, int(leadtime))
        if expedite:
            leadtime = max(1, leadtime - 2)
        self.order_pipeline.append((self.env.now + leadtime, qty))
        self.outstanding += qty

        self.env.step()
        self.current_step += 1

        arrivals = [q for t, q in self.order_pipeline if t <= self.env.now]
        self.inventory += sum(arrivals)
        self.outstanding -= sum(arrivals)
        self.order_pipeline = [(t, q) for t, q in self.order_pipeline if t > self.env.now]

        demand = self.demand_forecast
        fulfilled = min(self.inventory, demand)
        self.inventory -= fulfilled
        stockout = max(0, demand - fulfilled)

        self.scri = min(1.0, 0.5 * (stockout / (demand + 1e-6)) + 0.5 * (self.disruption / 2))

        holding_cost = 0.1 * self.inventory
        stockout_cost = 2.0 * stockout
        order_cost = 1.0 * qty
        disruption_cost = 5.0 * (self.disruption > 0)
        scri_penalty = 10.0 * (self.scri > 0.7)
        cost = holding_cost + stockout_cost + order_cost + disruption_cost + scri_penalty
        self.total_cost += cost

        self.done = self.current_step >= self.max_steps
        obs = self._get_obs()
        info = {"cost": cost, "scri": self.scri}
        return obs, -cost, self.done, info

    def _get_obs(self):
        return np.array([
            self.inventory,
            self.outstanding,
            self.leadtime,
            self.disruption,
            self.scri
        ], dtype=np.float32)

    def render(self, mode="human"):
        print(f"Step {self.current_step}: Inv={self.inventory}, Out={self.outstanding}, Disr={self.disruption}, SCRI={self.scri:.2f}")

    def close(self):
        pass