from utils.config import SUPABASE_URL, SUPABASE_KEY

_client = None


def get_client():
    """Create the Supabase client on first use, so offline runs never need it."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are not set")
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client
