from db.supabase_client import supabase

def fetch_emails():
    res = supabase.table("emails").select("*").execute()

    print("DATA:", res.data)

    return res.data

if __name__ == "__main__":
    fetch_emails()