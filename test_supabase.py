from supabase import create_client

url = "https://oiaifpdfuofguwvvdveu.supabase.co"
key = "sb_publishable_PpmqLENVhKaNsdvI9Ni2aQ_h77hG9cx"

supabase = create_client(url, key)

res = supabase.table("emails").select("*").execute()

print("DATA:", res.data)
print("ERROR:", res.error)