# tools/calendar.py

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os
import datetime


def load_credentials():
    token_path = "token.json"
    if not os.path.exists(token_path):
        raise Exception("❌ No token.json found. Authorize first.")
    creds = Credentials.from_authorized_user_file(token_path)
    return creds

def create_calendar_event(form_data):
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json")

    service = build("calendar", "v3", credentials=creds)

    # Default fallback values
    title = form_data.get("title", "Meeting with Alfred")
    datetime_str = form_data.get("datetime", datetime.datetime.utcnow().isoformat() + "Z")
    description = form_data.get("description", "Scheduled via GhostStack AI assistant.")

    event = {
        'summary': title,
        'description': description,
        'start': {'dateTime': datetime_str, 'timeZone': 'America/Chicago'},
        'end': {'dateTime': datetime_str, 'timeZone': 'America/Chicago'},  # ❗ temporary — no duration yet
    }

    created_event = service.events().insert(calendarId='primary', body=event).execute()
    return {"status": f"✅ Event created: {created_event.get('htmlLink')}"}

def store_token(token_json: str):
    with open("token.json", "w") as f:
        f.write(token_json)
