# tools/calendar.py

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os
import datetime
from langchain.tools import tool
from typing import Optional



def load_credentials():
    token_path = "token.json"
    if not os.path.exists(token_path):
        raise Exception("❌ No token.json found. Authorize first.")
    creds = Credentials.from_authorized_user_file(token_path)
    return creds

import os
import dateparser
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import timedelta

def create_calendar_event(
    datetime_str: Optional[str],
    name: Optional[str],
    business_name: Optional[str],
    address: Optional[str],
    phone: Optional[str],
    email: Optional[str]
) -> str:
    """
    Creates a real event in Google Calendar using parsed booking details.
    """
    # Validate required fields
    missing_fields = []
    if not datetime_str:
        missing_fields.append("datetime")
    if not name:
        missing_fields.append("name")
    if not business_name:
        missing_fields.append("business name")
    if not address:
        missing_fields.append("address")
    if not phone:
        missing_fields.append("phone")
    if not email:
        missing_fields.append("email")

    if missing_fields:
        print(f"❌ Missing fields for booking: {missing_fields}")
        return f"Unable to book your appointment — missing: {', '.join(missing_fields)}."

    print("📆 Starting real calendar booking...")

    # Parse datetime
    parsed_start = dateparser.parse(datetime_str)
    if not parsed_start:
        return "❌ I couldn't understand the date/time. Please rephrase it."

    parsed_end = parsed_start + timedelta(hours=1)

    try:
        token_path = "token.json"
        if not os.path.exists(token_path):
            return "❌ Calendar authorization token missing."

        creds = Credentials.from_authorized_user_file(token_path)
        service = build("calendar", "v3", credentials=creds)

        # Build event dynamically
        event = {
            'summary': f'Meeting with {name} from {business_name}',
            'description': f'Scheduled via Alfred.\n\nBusiness: {business_name}\nAddress: {address}\nPhone: {phone}\nEmail: {email}',
            'start': {
                'dateTime': parsed_start.isoformat(),
                'timeZone': 'America/Chicago',
            },
            'end': {
                'dateTime': parsed_end.isoformat(),
                'timeZone': 'America/Chicago',
            },
            'attendees': [
                {'email': email},
            ],
        }

        created_event = service.events().insert(calendarId='primary', body=event).execute()
        event_link = created_event.get('htmlLink')

        print(f"✅ Event created: {event_link}")
        return f"✅ All set, {name}!\n\n📅 Your appointment is scheduled for **{parsed_start.strftime('%A, %B %-d at %-I:%M %p')}**.\nWe’ve also added it to our calendar: {event_link}"

    except Exception as e:
        print(f"❌ Calendar error: {e}")
        return f"❌ Failed to book your appointment: {str(e)}"



def store_token(token_json: str):
    with open("token.json", "w") as f:
        f.write(token_json)


# def create_calendar_event(
#     datetime_str: Optional[str],
#     name: Optional[str],
#     business_name: Optional[str],
#     address: Optional[str],
#     phone: Optional[str],
#     email: Optional[str]
# ) -> str:
#     """Simulates creating a calendar event and returns a confirmation message."""

#     # Validate required fields
#     missing_fields = []
#     if not datetime_str:
#         missing_fields.append("datetime")
#     if not name:
#         missing_fields.append("name")
#     if not business_name:
#         missing_fields.append("business name")
#     if not address:
#         missing_fields.append("address")
#     if not (phone or email):
#         missing_fields.append("phone or email")

#     if missing_fields:
#         print(f"❌ Missing fields for booking: {missing_fields}")
#         return f"Unable to book your appointment — missing: {', '.join(missing_fields)}."

#     print(f"✅ Booking event for {name} at {datetime_str}")
    
#     contact = phone if phone else email

#     return (
#         f"✅ All set, {name}!\n\n"
#         f"📅 Your call is booked for **{datetime_str}**.\n"
#         f"🏢 Business: {business_name}\n"
#         f"📍 Address: {address}\n"
#         f"📞 Contact: {contact}\n\n"
#         f"We’ll send a quick reminder one hour before your call."
#     )
