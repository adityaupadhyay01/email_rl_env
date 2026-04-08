import os
from fastapi import FastAPI
from env.email_env import EmailEnv
from agent.baseline_agent import BaselineAgent

# ENV VARS (required)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-model")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

# Global env state (IMPORTANT for OpenEnv)
env = None
state = None


# HEALTH CHECK (root)
@app.get("/")
def home():
    return {"message": "Email RL Agent Running 🚀"}


# RESET (MANDATORY for OpenEnv)
@app.post("/reset")
def reset():
    global env, state

    env = EmailEnv()
    state = env.reset()

    return {"state": state}


# STEP (MANDATORY for OpenEnv)
@app.post("/step")
def step(action: dict):
    global env, state

    action_value = action.get("action")

    state, reward, done = env.step(action_value)

    return {
        "state": state,
        "reward": reward,
        "done": done
    }


# OPTIONAL: manual run (for browser testing)
@app.get("/run")
def run_agent():
    env_local = EmailEnv()
    agent = BaselineAgent()

    state_local = env_local.reset()
    done = False
    logs = []
    step_count = 0
    total_reward = 0

    while not done:
        action = agent.act(state_local)
        state_local, reward, done = env_local.step(action)

        logs.append({
            "step": step_count,
            "action": action,
            "reward": reward
        })

        total_reward += reward
        step_count += 1

    score = max(0, min(1, (total_reward / step_count + 1) / 2)) if step_count > 0 else 0

    return {
        "steps": step_count,
        "score": score,
        "logs": logs
    }


# CLI RUN (IMPORTANT for logs format)
def run():
    env_local = EmailEnv()
    agent = BaselineAgent()

    rewards = []
    total_reward = 0

    print("[START] task=email env=email_rl model=baseline")

    state_local = env_local.reset()

    for step in range(1, len(env_local.data) + 1):
        action = agent.act(state_local)
        state_local, reward, done = env_local.step(action)

        rewards.append(reward)
        total_reward += reward

        print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

        if done:
            break

    score = max(0, min(1, (total_reward / len(rewards) + 1) / 2)) if rewards else 0
    success = score > 0.5

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    run()
