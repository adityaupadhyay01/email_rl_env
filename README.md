# Email RL Environment — Intelligent Spam Classification System

## Team: **Kairos**

## Project Overview

This project implements a **Reinforcement Learning (RL) based Email Classification System** that intelligently categorizes emails into:
* Spam
* Not Spam

Unlike traditional classifiers, this system uses a **reward-driven learning environment**, simulating decision-making similar to real-world intelligent agents.

## Key Highlights

* Custom Reinforcement Learning Environment (`EmailEnv`)
* Baseline Agent (rule-based decision making)
* Reward shaping for improved learning
* Supabase integration for real-time data
* Fallback dataset support (offline mode)
* Fully Dockerized (production-ready)
* Deployed on HuggingFace Spaces

---

## Project Architecture

```
email-rl-env/
│
├── env/            # RL Environment logic
├── agent/          # Baseline & Policy agents
├── db/             # Supabase connection
├── data/           # Fallback JSON dataset
├── evaluator/      # Performance grading
├── utils/          # Helper utilities
│
├── inference.py    # Main execution pipeline
├── Dockerfile      # Deployment config
├── requirements.txt
└── README.md
```

## How It Works

1. Emails are fetched from **Supabase**
2. The agent analyzes email text
3. Takes an action: `spam` or `not_spam`
4. Environment returns a reward based on correctness
5. Process repeats until all emails are processed

## Reward Strategy

| Scenario                     | Reward |
| ---------------------------- | ------ |
| Correct classification       | +1     |
| Incorrect classification     | -0.5   |
| Spam keyword detection bonus | +0.5   |
| False spam detection penalty | -0.5   |

This ensures smarter and more balanced learning.


##  Sample Output

```
[START] task=email env=email_rl model=baseline
[STEP] step=1 action=spam reward=1.50 done=false
...
[END] success=true steps=15 score=1.00
```

## Run Locally with Docker

```bash
docker build -t email-env .
docker run email-env
```

## Environment Variables

```env
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_api_key
```

> Note: Use secure environment variables (do NOT hardcode keys)

## Deployment

This project is successfully deployed using:

**HuggingFace Spaces (Docker-based deployment)**

## Why This Project Stands Out

* Goes beyond static ML → uses **decision-based learning**
* Simulates real-world agent behavior
* Production-ready with Docker
* Handles both online (Supabase) and offline (fallback JSON) modes
* Clean modular architecture

## Future Improvements

*  Train a Policy-based RL agent
*  Add performance analytics dashboard
*  Integrate LLM-based classification
*  Real-time email streaming

## Team

Aditya Upadhyay
Shreya Pathak
Anurag Mishra

## Final Note
This project demonstrates the power of combining **Reinforcement Learning + Backend Systems + Deployment**, creating a scalable and intelligent email classification system.

*Built with precision. Deployed with confidence.*
