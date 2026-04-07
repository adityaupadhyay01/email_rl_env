def process_state(email):
    return {
        "text": email["text"],
        "length": len(email["text"]),
        "has_offer": "free" in email["text"].lower()
    }