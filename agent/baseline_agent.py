import random

class RandomAgent:
    def __init__(self):
        self.actions = [
            "mark_spam",
            "work",
            "personal",
            "promotion"
        ]

    def act(self, state):
        return random.choice(self.actions)