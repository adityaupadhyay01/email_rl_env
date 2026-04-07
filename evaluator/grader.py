def evaluate(agent, env, episodes=50):
    total_reward = 0

    for _ in range(episodes):
        state = env.reset()
        action = agent.act(state)
        _, reward, _, _ = env.step(action)
        total_reward += reward

    score = total_reward / episodes

    # normalize to 0–1
    score = max(0, min(1, (score + 1) / 3))

    return score