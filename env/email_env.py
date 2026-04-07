import random
from db.fetch_data import fetch_emails
from env.state_processor import process_state

class EmailEnv:
    def __init__(self):
        self.data = fetch_emails()
        self.current = None

    def reset(self):
        self.current = random.choice(self.data)
        return process_state(self.current)

    def step(self, action):
        reward = 0
        done = True

        # spam detection
        if action == "mark_spam":
            if self.current["label"] == "spam":
                reward += 1
            else:
                reward -= 1

        # category
        elif action == self.current["category"]:
            reward += 2
        else:
            reward -= 1

        return process_state(self.current), reward, done, {}