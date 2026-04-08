import os

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-model")
HF_TOKEN = os.getenv("HF_TOKEN")

from env.email_env import EmailEnv
from agent.baseline_agent import BaselineAgent

def run():
    env = EmailEnv()
    agent = BaselineAgent()

    rewards = []
    total_reward = 0

    print("[START] task=email env=email_rl model=baseline")

    state = env.reset()


    for step in range(1, len(env.data) + 1):
        
        action = agent.act(state)

        next_state, reward, done = env.step(action)

        rewards.append(reward)
        total_reward += reward

        print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

        state = next_state

        if done:
            break

    score = max(0, min(1, (total_reward / len(rewards) + 1) / 2)) if rewards else 0
    success = score > 0.5

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    run()