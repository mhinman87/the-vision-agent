# tools/calendar.py

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os
from datetime import datetime, timedelta

def load_credentials():
    token_path = "token.json"
    if not os.path.exists(token_path):
        raise Exception("❌ No token.json found. Authorize first.")
    creds = Credentials.from_authorized_user_file(token_path)
    return creds

def create_calendar_event(title, start_time_str):
    creds = load_credentials()
    service = build("calendar", "v3", credentials=creds)

    # Parse start time
    start_time = datetime.fromisoformat(start_time_str)
    end_time = start_time + timedelta(minutes=30)

    event = {
        'summary': title,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'America/Chicago',  # Or use pytz if needed
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'America/Chicago',
        },
    }

    created_event = service.events().insert(calendarId='primary', body=event).execute()
    return {"status": f"✅ Event created: {created_event.get('htmlLink')}"}

