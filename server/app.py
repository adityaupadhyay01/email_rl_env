from fastapi import FastAPI
from env.email_env import EmailEnv
from agent.baseline_agent import BaselineAgent

app = FastAPI()

env = None
state = None

@app.post("/reset")
def reset():
    global env, state
    env = EmailEnv()
    state = env.reset()
    return {"state": state}

@app.post("/step")
def step(action: dict):
    global env, state
    action_value = action.get("action")
    state, reward, done = env.step(action_value)
    return {"state": state, "reward": reward, "done": done}

    def main():
    return app
