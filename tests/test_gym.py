"""The Gymnasium wrapper follows the Gymnasium API."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

from db.fetch_data import load_fallback  # noqa: E402
from env.gym_env import EmailTriageEnv  # noqa: E402


def make():
    return EmailTriageEnv(emails=load_fallback())


def test_passes_the_gymnasium_env_checker():
    check_env(make(), skip_render_check=True)


def test_episode_is_one_email_and_rewards_match():
    env = make()
    obs, info = env.reset(seed=1, options={"index": 14})       # E15, the KYC scam
    assert obs.shape == env.observation_space.shape and env.observation_space.contains(obs)
    assert info["email_id"] == "E15" and info["action_mask"].tolist() == [1, 1, 1, 1, 1, 1]
    obs, r1, term, trunc, info = env.step(0)                    # check_content
    assert r1 == -0.05 and not term and info["action_mask"][0] == 0
    obs, r2, term, trunc, info = env.step(3)                    # quarantine
    assert r2 == 1.0 and term and not trunc and info["outcome"] == "correct"


def test_random_policy_runs_and_label_never_leaks():
    env = make()
    rng = np.random.default_rng(0)
    for episode in range(20):
        obs, info = env.reset(seed=episode)
        for _ in range(10):
            obs, r, term, trunc, info = env.step(int(rng.integers(6)))
            if term:
                break
        assert term
    assert "label" not in str(env._state)
