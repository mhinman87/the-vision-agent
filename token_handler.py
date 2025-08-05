import os
import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_persistent_credentials():
    token_json = os.getenv("GOOGLE_TOKEN")
    
    if not token_json:
        print("❌ No GOOGLE_TOKEN found in env vars")
        return None
    
    print("✅ GOOGLE_TOKEN found")

    try:
        creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)
        if creds.expired and creds.refresh_token:
            print("🔁 Refreshing token...")
            creds.refresh(Request())
            print("✅ Token refreshed")

        return creds
    except Exception as e:
        print("🔥 Error loading credentials:", str(e))
        return None
