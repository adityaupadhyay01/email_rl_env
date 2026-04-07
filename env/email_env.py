def step(self, action):
    reward = 0
    done = True

    true_label = self.current["label"]
    text = self.current["text"].lower()

    # Correct spam detection
    if action == "spam" and true_label == "spam":
        reward += 1

    elif action == "not_spam" and true_label == "not_spam":
        reward += 1

    else:
        reward -= 0.5

    # Bonus: keyword match
    if "free" in text or "win" in text:
        if action == "spam":
            reward += 0.5

    # Penalty for wrong spam
    if action == "spam" and true_label != "spam":
        reward -= 0.5

    return self._get_state(), reward, done