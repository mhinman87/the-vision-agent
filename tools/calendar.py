# tools/calendar.py

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os
import datetime
from langchain.tools import tool


def load_credentials():
    token_path = "token.json"
    if not os.path.exists(token_path):
        raise Exception("❌ No token.json found. Authorize first.")
    creds = Credentials.from_authorized_user_file(token_path)
    return creds

@tool
def create_calendar_event(
    title: str = "Meeting with Alfred",
    datetime: str = "2025-08-01T14:00:00",
    description: str = "Scheduled via GhostStack AI assistant."
) -> str:
    """
    Add an event to the GhostStack company calendar.

    Args:
        title (str): The title of the meeting.
        datetime (str): The ISO-formatted datetime string (e.g. "2025-08-01T14:00:00").
        description (str): Description of the event.

    Returns:
        str: A success message with the calendar event link.
    """

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json")

    service = build("calendar", "v3", credentials=creds)

    # Default fallback values

    event = {
        'summary': title,
        'description': description,
        'start': {'dateTime': datetime, 'timeZone': 'America/Chicago'},
        'end': {'dateTime': datetime, 'timeZone': 'America/Chicago'},  # Can be improved later
    }

    created_event = service.events().insert(calendarId='primary', body=event).execute()
    return f"✅ Event created: {created_event.get('htmlLink')}"


def store_token(token_json: str):
    with open("token.json", "w") as f:
        f.write(token_json)
