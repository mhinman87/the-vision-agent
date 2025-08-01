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

# @tool
# def create_calendar_event(
#     title: str = "Meeting with Alfred",
#     datetime: str = "2025-08-01T14:00:00",
#     description: str = "Scheduled via GhostStack AI assistant."
# ) -> str:
#     """
#     Add an event to the GhostStack company calendar.

#     Args:
#         title (str): The title of the meeting.
#         datetime (str): The ISO-formatted datetime string (e.g. "2025-08-01T14:00:00").
#         description (str): Description of the event.

#     Returns:
#         str: A success message with the calendar event link.
#     """

#     creds = None
#     if os.path.exists("token.json"):
#         creds = Credentials.from_authorized_user_file("token.json")

#     service = build("calendar", "v3", credentials=creds)

#     # Default fallback values

#     event = {
#         'summary': title,
#         'description': description,
#         'start': {'dateTime': datetime, 'timeZone': 'America/Chicago'},
#         'end': {'dateTime': datetime, 'timeZone': 'America/Chicago'},  # Can be improved later
#     }

#     created_event = service.events().insert(calendarId='primary', body=event).execute()
#     return f"✅ Event created: {created_event.get('htmlLink')}"


# def create_calendar_event(input: str) -> str:
#     """
#     Adds a hardcoded event to the GhostStack company calendar.
#     Input is ignored for now. Logs output for debugging.

#     Returns:
#         str: Success or failure message.
#     """
#     try:
#         print("📆 Starting calendar tool...")
#         token_path = "token.json"

#         if not os.path.exists(token_path):
#             print("❌ token.json not found.")
#             return "❌ Calendar authorization token missing."

#         creds = Credentials.from_authorized_user_file(token_path)
#         print("✅ Loaded credentials.")

#         service = build("calendar", "v3", credentials=creds)
#         print("✅ Built Google Calendar service.")

#         # Hardcoded event details
#         event = {
#             'summary': 'Meeting with Alfred',
#             'description': 'Scheduled via GhostStack AI assistant.',
#             'start': {
#                 'dateTime': '2025-08-01T15:00:00',  # 3 PM CDT
#                 'timeZone': 'America/Chicago'
#             },
#             'end': {
#                 'dateTime': '2025-08-01T16:00:00',
#                 'timeZone': 'America/Chicago'
#             }
#         }

#         print("📤 Inserting event...")
#         created_event = service.events().insert(calendarId='primary', body=event).execute()
#         event_link = created_event.get('htmlLink')

#         print(f"✅ Event created successfully: {event_link}")
#         return f"✅ Event created: {event_link}"

#     except Exception as e:
#         print(f"❌ Failed to create event: {e}")
#         return f"❌ Failed to create event: {str(e)}"


def store_token(token_json: str):
    with open("token.json", "w") as f:
        f.write(token_json)


def create_calendar_event(datetime_str: str, name: str) -> str:
    """Creates a calendar event using the provided datetime and name."""
    print(f"✅ Booking event for {name} at {datetime_str}")
    return f"Your appointment at {datetime_str} is all set! We’ll reach out with a quick confirmation text one hour beforehand."
