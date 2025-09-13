import pytest
import numpy as np
from src.env_supplychain import SupplyChainSimEnv

def test_env_reset_and_step_consistency():
    env = SupplyChainSimEnv(seed=42)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    for _ in range(5):
        action = {"order_qty": 5, "expedite": 0, "mitigate": 0}
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "cost" in info
        if done:
            break

def test_env_reproducibility():
    env1 = SupplyChainSimEnv(seed=123)
    env2 = SupplyChainSimEnv(seed=123)
    obs1 = env1.reset()
    obs2 = env2.reset()
    np.testing.assert_allclose(obs1, obs2)
    for _ in range(10):
        action = {"order_qty": 3, "expedite": 1, "mitigate": 0}
        o1, r1, d1, i1 = env1.step(action)
        o2, r2, d2, i2 = env2.step(action)
        np.testing.assert_allclose(o1, o2)
        assert r1 == pytest.approx(r2)
        assert d1 == d2

def test_env_constraints():
    env = SupplyChainSimEnv(seed=7)
    obs = env.reset()
    for _ in range(3):
        action = {"order_qty": 20, "expedite": 1, "mitigate": 1}
        obs, reward, done, info = env.step(action)
        assert obs[0] >= 0
        assert obs[1] >= 0
        assert 0.0 <= obs[4] <= 1.0

def test_env_done_flag():
    env = SupplyChainSimEnv(seed=99, config={"max_steps": 5})
    env.reset()
    for i in range(5):
        action = {"order_qty": 1, "expedite": 0, "mitigate": 0}
        obs, reward, done, info = env.step(action)
        if i < 4:
            assert not done
        else:
            assert done