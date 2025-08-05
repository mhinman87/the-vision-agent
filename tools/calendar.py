# tools/calendar.py

import os
import dateparser
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import timedelta
from typing import Optional
from token_handler import get_persistent_credentials

from openai import OpenAI

client = OpenAI()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import datetime

def parse_datetime_with_llm(natural_str: str) -> Optional[datetime.datetime]:
    today = datetime.datetime.now().strftime("%A, %B %d, %Y")
    print(f"🧠 Asking LLM to convert '{natural_str}' based on today: {today}")

    system_prompt = (
        f"You are a precise date parser. Today is {today}.\n"
        "Convert the provided natural language date/time into an ISO 8601 datetime string "
        "(e.g. 2025-08-07T17:00:00). Return only the ISO string. Do not include any explanation."
    )

    llm = ChatOpenAI(model="gpt-4o")

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=natural_str)
    ])

    clean_str = response.content.strip()  # ✅ This is the fix

    try:
        parsed = datetime.datetime.fromisoformat(clean_str)
        return parsed
    except Exception as e:
        print(f"❌ LLM returned invalid ISO: {clean_str} — {e}")
        return None





def load_credentials():
    creds = get_persistent_credentials()
    if not creds:
        raise Exception("❌ No calendar credentials found — authorization required.")
    return creds



from token_handler import get_persistent_credentials
from googleapiclient.discovery import build
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
    parsed_start = parse_datetime_with_llm(datetime_str)
    if not parsed_start:
        return "❌ I couldn't understand the date/time. Please rephrase it."

    parsed_end = parsed_start + timedelta(hours=1)

    # ✅ Use persistent credentials
    creds = get_persistent_credentials()
    if not creds:
        print("❌ No credentials available")
        return "❌ Calendar authorization token missing."

    try:
        service = build("calendar", "v3", credentials=creds)

        # Build event
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

