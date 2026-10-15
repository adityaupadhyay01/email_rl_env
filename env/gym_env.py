"""
Gymnasium interface to the email threat-triage environment.

    import gymnasium as gym
    from env.gym_env import EmailTriageEnv

    env = EmailTriageEnv(split="train")          # or emails=[...]
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

One episode = one email. The agent may investigate (check_content, check_sender,
check_url) and then must decide (quarantine, allow, escalate); the decision ends the
episode. Rewards are exactly those of EmailEnv.

Observation: a float vector in [0, 1] (agent.pg_agent.features): for each check whether
it ran, its risk, whether it had nothing to check, how much of its evidence was
unavailable, whether the risk is >= 0.8; then max and mean risk, has links, has
attachments, steps left, bias. The label is never part of it.

info["action_mask"] marks the actions that make sense (no repeated checks, no URL
check without links), for algorithms that support masking. Unmasked actions are still
legal: a repeated check just costs reward again.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent.pg_agent import ACTIONS, N_FEATURES, action_mask, features
from email_env import EmailEnv


class EmailTriageEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, emails=None, split="train", tools=None, render_mode=None,
                 false_positive_cost=1.0, escalate_reward=0.3):
        if emails is None:
            from evaluator.datasets import load_split
            emails = load_split(split)
        if not emails:
            raise ValueError("EmailTriageEnv needs at least one email")
        if tools is None:
            from env.tools import InvestigationTools
            tools = InvestigationTools()
        self.emails = list(emails)
        self.tools = tools
        self.render_mode = render_mode
        self.costs = {"false_positive_cost": false_positive_cost,
                      "escalate_reward": escalate_reward}
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_FEATURES,), dtype=np.float32)
        self._env = None
        self._state = None
        self._last = ""

    def _obs(self):
        return features(self._state).astype(np.float32)

    def _info(self, extra=None):
        info = {"action_mask": action_mask(self._state).astype(np.int8),
                "email_id": self._state.get("id")}
        info.update(extra or {})
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("index")
        if index is None:
            index = int(self.np_random.integers(len(self.emails)))
        self._env = EmailEnv(emails=[self.emails[index]], tools=self.tools, **self.costs)
        self._state = self._env.reset()
        self._last = f"email {self._state['id']}"
        return self._obs(), self._info()

    def step(self, action):
        if self._env is None or self._env.done:
            raise RuntimeError("Call reset() before step()")
        name = ACTIONS[int(action)]
        state, reward, done, info = self._env.step(name)
        self._last = f"{name}: reward {reward:+.2f}"
        if done:
            # Keep the last observation for the terminal step.
            obs = self._obs()
            return obs, float(reward), True, False, {
                "action_mask": np.zeros(len(ACTIONS), dtype=np.int8),
                "email_id": info["email_id"], "outcome": info["outcome"],
                "truth": info["truth"], "evidence": info["evidence"]}
        self._state = state
        extra = {"evidence": info.get("evidence")} if info["kind"] == "investigate" else {}
        return self._obs(), float(reward), False, False, self._info(extra)

    def render(self):
        if self.render_mode == "ansi":
            return self._last
        return None
