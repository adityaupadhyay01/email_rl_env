from env.email_env import EmailEnv
from agent.baseline_agent import RandomAgent
from evaluator.grader import evaluate

def main():
    env = EmailEnv()
    agent = RandomAgent()

    score = evaluate(agent, env)

    print(f"Score: {score:.2f}")

if __name__ == "__main__":
    main()