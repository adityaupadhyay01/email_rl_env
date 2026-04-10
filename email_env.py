from db.fetch_data import fetch_emails


class EmailEnv:
    def __init__(self):
        self.data = fetch_emails()
        self.index = 0
        self.current = None

    def reset(self):
        self.index = 0
        self.current = self.data[self.index]
        return self._get_state()

    def _get_state(self):
        if self.index < len(self.data):
            self.current = self.data[self.index]
            return self.current
        return None

    def step(self, action):
        reward = 0

        true_label = self.current["label"]
        text = self.current["text"].lower()

        # Correct classification
        if action == true_label:
            reward += 1
        else:
            reward -= 0.5

        # Bonus for smart detection
        spam_keywords = ["free", "win", "offer", "prize", "discount"]

        if any(word in text for word in spam_keywords):
            if action == "spam":
                reward += 0.5

        # Extra penalty (false spam)
        if action == "spam" and true_label != "spam":
            reward -= 0.5

        # Move to next email
        self.index += 1

        # Done condition
        done = self.index >= len(self.data)

        return self._get_state(), reward, done