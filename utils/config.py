import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:      # python-dotenv is optional when running offline
    pass

# Where emails come from: "local" (data/fallback.json) or "supabase".
EMAIL_SOURCE = os.getenv("EMAIL_SOURCE", "local").strip().lower()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# LLM agent (any OpenAI-compatible endpoint; HuggingFace router by default).
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
