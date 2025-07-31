# tools/calendar.py

import os
import json
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# In-memory token store (upgrade to persistent storage later)
user_tokens = {}

# Use this to manually add the token from your /oauth2callback
def store_token(token_json: str):
    user_tokens["default"] = token_json

def is_authorized():
    return "default" in user_tokens

def create_calendar_event(summary="GhostStack Call", start_time=None, duration_minutes=30):
    if "default" not in user_tokens:
        return "User is not authorized. Please visit /auth to connect your calendar."

    creds = Credentials.from_authorized_user_info(json.loads(user_tokens["default"]))
    service = build("calendar", "v3", credentials=creds)

    if not start_time:
        start_time = datetime.utcnow() + timedelta(days=1)
    end_time = start_time + timedelta(minutes=duration_minutes)

    event = {
        "summary": summary,
        "start": {"dateTime": start_time.isoformat() + "Z", "timeZone": "UTC"},
        "end": {"dateTime": end_time.isoformat() + "Z", "timeZone": "UTC"},
    }

    created_event = service.events().insert(calendarId="primary", body=event).execute()
    return f"📅 Event created: {created_event.get('htmlLink')}"
