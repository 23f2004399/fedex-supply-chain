import pytest
import numpy as np
from src.env_supplychain import SupplyChainSimEnv, StudentTCopulaSampler
from src.agent_qlearning import QLearningAgent, get_action_space, get_bins, discretize_obs
from scipy.stats import norm

def test_env_determinism():
    seed = 666
    env1 = SupplyChainSimEnv(seed=seed)
    env2 = SupplyChainSimEnv(seed=seed)

    initial_obs1 = env1.reset()
    initial_obs2 = env2.reset()

    np.testing.assert_allclose(initial_obs1, initial_obs2, err_msg="Initial observations should be identical")

    actions_to_perform = [
        {"order_qty": 18, "expedite": 0, "mitigate": 1},
        {"order_qty": 5, "expedite": 1, "mitigate": 0}, 
        {"order_qty": 10, "expedite": 0, "mitigate": 0},
    ]

    for action in actions_to_perform:
        obs1, reward1, done1, i1 = env1.step(action)
        obs2, reward2, done2, i2 = env2.step(action)

        np.testing.assert_allclose(obs1, obs2, err_msg=f"Observations for action {action} should match")
        assert pytest.approx(reward1) == reward2, f"Rewards for action {action} should match"
        assert done1 == done2, f"Done flags for action {action} should match"



def test_q_learning_update_maintains_shape():
    obs_bins = get_bins()
    action_space = get_action_space()
    agent = QLearningAgent(obs_bins, action_space)
    num_actions = len(action_space)

    current_obs = np.array([50, 10, 5, 0, 0.1])
    next_obs = np.array([40, 20, 5, 0, 0.2])   
    action_index = 3                           
    reward = -10.5                             
    done = False 

    state_key = discretize_obs(current_obs, obs_bins)

    initial_q_values = agent.q_table[state_key]
    assert initial_q_values.shape == (num_actions,), "Initial Q-value array shape should match the number of actions"

    agent.update(current_obs, action_index, reward, next_obs, done)

    updated_q_values = agent.q_table[state_key]
    assert updated_q_values.shape == (num_actions,), "Q-value array shape should NOT change after an update"

    assert not np.all(updated_q_values == 0), "The update function should change the Q-values"


def test_sampler_reproducibility():
    marginals = [norm(loc=10, scale=3), norm(loc=50, scale=10)]
    correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    seed = 42

    sampler1 = StudentTCopulaSampler(marginals, correlation_matrix, seed=seed)
    sampler2 = StudentTCopulaSampler(marginals, correlation_matrix, seed=seed)

    num_samples_to_generate = 20
    generated_samples1 = sampler1.sample(n=num_samples_to_generate)
    generated_samples2 = sampler2.sample(n=num_samples_to_generate)

    assert generated_samples1.shape == (num_samples_to_generate, len(marginals))

    np.testing.assert_allclose(
        generated_samples1,
        generated_samples2,
        err_msg="Samplers with the same seed should produce identical output"
    )