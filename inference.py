import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI

try:
    from email_env import EmailEnv
from baseline_agent import BaselineAgent
except Exception as e:
    print("[IMPORT ERROR]", str(e))
    raise e

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-model")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

env = None
state = None

@app.get("/")
def home():
    return {"message": "Email RL Agent Running 🚀"}

@app.post("/reset")
def reset():
    global env, state
    try:
        env = EmailEnv()
        state = env.reset()
        return {"state": state}
    except Exception as e:
        print("[RESET ERROR]", str(e))
        return {"error": str(e)}

@app.post("/step")
def step(action: dict):
    global env, state
    try:
        action_value = action.get("action")
        state, reward, done = env.step(action_value)
        return {
            "state": state,
            "reward": reward,
            "done": done
        }
    except Exception as e:
        print("[STEP ERROR]", str(e))
        return {"error": str(e)}


# OPTIONAL TEST ENDPOINT
@app.get("/run")
def run_agent():
    try:
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

    except Exception as e:
        print("[RUN ERROR]", str(e))
        return {"error": str(e)}

def run():
    try:
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

    except Exception as e:
        print("[FATAL ERROR]", str(e))

if __name__ == "__main__":
    run()
