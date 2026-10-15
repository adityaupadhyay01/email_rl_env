class BaselineAgent:
    def act(self, state):
        text = state["text"].lower()

        spam_keywords = ["free", "win", "offer", "prize", "discount", "lottery"]

        if any(word in text for word in spam_keywords):
            return "spam"

        return "not_spam"