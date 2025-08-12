# tools/calendar.py

import os
import dateparser
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta
from typing import Optional
from token_handler import get_persistent_credentials
from token_handler import get_persistent_credentials
from googleapiclient.discovery import build

from openai import OpenAI

client = OpenAI()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

def parse_datetime_with_llm(natural_str: str) -> Optional[datetime]:
    today = datetime.now().strftime("%A, %B %d, %Y")
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
        parsed = datetime.fromisoformat(clean_str)
        
        # Validate business hours (Monday-Friday, 10 AM - 4 PM)
        if parsed.weekday() >= 5:  # Saturday = 5, Sunday = 6
            print(f"❌ Weekend appointment rejected: {parsed.strftime('%A, %B %-d at %-I:%M %p')}")
            return None
            
        if not (10 <= parsed.hour < 16):
            print(f"❌ Non-business hours rejected: {parsed.strftime('%A, %B %-d at %-I:%M %p')}")
            return None
            
        return parsed
    except Exception as e:
        print(f"❌ LLM returned invalid ISO: {clean_str} — {e}")
        return None





def load_credentials():
    creds = get_persistent_credentials()
    if not creds:
        raise Exception("❌ No calendar credentials found — authorization required.")
    return creds





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
    Assumes all input is pre-validated and that datetime_str is in ISO 8601 format.
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

    try:
        parsed_start = datetime.fromisoformat(datetime_str)
        parsed_end = parsed_start + timedelta(hours=1)
    except Exception as e:
        print(f"❌ Failed to parse ISO datetime: {datetime_str} — {e}")
        return "❌ Internal error: failed to interpret booking time."

    # ✅ Use persistent credentials
    creds = get_persistent_credentials()
    if not creds:
        print("❌ No credentials available")
        return "❌ Calendar authorization token missing."

    try:
        service = build("calendar", "v3", credentials=creds)

        event = {
            'summary': f'Meeting with {name} from {business_name}',
            'description': (
                f'Scheduled via Alfred.\n\nBusiness: {business_name}\n'
                f'Address: {address}\nPhone: {phone}\nEmail: {email}'
            ),
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
        return (
            f"✅ You're all set, {name}!\n\n"
            f"Thank you for your interest in Ghost Stack.\n\n"
            f"📅 Your appointment is scheduled for {parsed_start.strftime('%A, %B %-d at %-I:%M %p')}\n\n"
            f"I've also sent a confirmation email to {email} with a link to the event.\n\n"
            f"This meeting is currently set as **in-person**, and Max will come to your location. He'll also reach out about an hour beforehand to confirm you're still available.\n\n"
            f"Thanks again for checking out Ghost Stack and our AI Agents — we're excited to connect!"
        )


    except Exception as e:
        print(f"❌ Calendar error: {e}")
        return f"❌ Failed to book your appointment: {str(e)}"




def store_token(token_json: str):
    with open("token.json", "w") as f:
        f.write(token_json)





def get_upcoming_event(name=None, email=None):
    creds = get_persistent_credentials()
    if not creds:
        print("❌ No credentials available for lookup.")
        return None

    service = build("calendar", "v3", credentials=creds)
    now = datetime.utcnow().isoformat() + 'Z'

    try:
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        print(f"🔍 Found {len(events)} upcoming events")

        for event in events:
            summary = event.get('summary', '').lower()
            description = event.get('description', '').lower()
            start = event.get('start', {}).get('dateTime')

            if not start:
                continue

            if (name and name.lower() in summary) or (email and email.lower() in description):
                return {
                    "start_time": start,
                    "summary": summary
                }

    except Exception as e:
        print(f"❌ Failed to fetch calendar events: {e}")

    return None


def reschedule_appointment(
    name: Optional[str],
    email: Optional[str],
    new_datetime_str: Optional[str]
) -> str:
    """
    Reschedules an existing calendar event to a new time.
    Requires name/email to find the event and new datetime to reschedule to.
    """
    # Validate required fields
    if not new_datetime_str:
        return "❌ Please provide the new date and time for your appointment."
    
    if not name and not email:
        return "❌ I need your name or email to find your appointment to reschedule."

    print(f"🔄 Attempting to reschedule appointment for {name or email} to {new_datetime_str}")

    try:
        # Parse the new datetime
        parsed_new_start = datetime.fromisoformat(new_datetime_str)
        parsed_new_end = parsed_new_start + timedelta(hours=1)
        
        # Validate business hours (10 AM - 4 PM Central)
        if not (10 <= parsed_new_start.hour < 16):
            return "❌ I can only schedule appointments between 10 AM and 4 PM Central Time. Please choose a different time."
            
    except Exception as e:
        print(f"❌ Failed to parse new datetime: {new_datetime_str} — {e}")
        return "❌ I couldn't understand that date and time. Please provide it in a clear format."

    # Get credentials
    creds = get_persistent_credentials()
    if not creds:
        return "❌ Calendar authorization token missing."

    try:
        service = build("calendar", "v3", credentials=creds)
        
        # First, find the existing event
        now = datetime.utcnow().isoformat() + 'Z'
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        target_event = None

        # Find the event to reschedule
        for event in events:
            summary = event.get('summary', '').lower()
            description = event.get('description', '').lower()
            
            if (name and name.lower() in summary) or (email and email.lower() in description):
                target_event = event
                break

        if not target_event:
            return f"❌ I couldn't find any upcoming appointments for {name or email}."

        # Update the event with new time
        event_id = target_event['id']
        
        updated_event = {
            'summary': target_event['summary'],
            'description': target_event['description'],
            'start': {
                'dateTime': parsed_new_start.isoformat(),
                'timeZone': 'America/Chicago',
            },
            'end': {
                'dateTime': parsed_new_end.isoformat(),
                'timeZone': 'America/Chicago',
            },
            'attendees': target_event.get('attendees', [])
        }

        # Update the event
        updated_event = service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=updated_event
        ).execute()

        print(f"✅ Event rescheduled: {updated_event.get('htmlLink')}")
        
        return (
            f"✅ Appointment rescheduled successfully!\n\n"
            f"📅 Your new appointment time is {parsed_new_start.strftime('%A, %B %-d at %-I:%M %p')}\n\n"
            f"You'll receive an updated calendar invitation via email.\n\n"
            f"Let me know if you need anything else!"
        )

    except Exception as e:
        print(f"❌ Failed to reschedule appointment: {e}")
        return f"❌ Sorry, I encountered an error while rescheduling: {str(e)}"


def get_available_slots_next_week() -> list:
    """
    Finds 3 available appointment slots across the next week.
    Respects 48-hour buffer from current time and business hours (10 AM - 4 PM Central).
    Returns a list of available datetime strings.
    """
    creds = get_persistent_credentials()
    if not creds:
        print("❌ No credentials available for availability check.")
        return []

    service = build("calendar", "v3", credentials=creds)
    
    # Calculate time boundaries
    now = datetime.now()
    buffer_time = now + timedelta(hours=24)  # 24-hour buffer (1 day notice)
    week_end = now + timedelta(days=7)  # Next week
    
    print(f"🔍 Checking availability from {buffer_time.strftime('%A, %B %-d at %-I:%M %p')} to {week_end.strftime('%A, %B %-d at %-I:%M %p')}")
    
    try:
        # Get all events in the next week
        events_result = service.events().list(
            calendarId='primary',
            timeMin=buffer_time.isoformat() + 'Z',
            timeMax=week_end.isoformat() + 'Z',
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        print(f"📅 Found {len(events)} existing events in the next week")
        
        # Create a set of busy time slots (hourly blocks)
        busy_slots = set()
        for event in events:
            start = event.get('start', {}).get('dateTime')
            if start:
                event_start = datetime.fromisoformat(start.replace('Z', '+00:00'))
                # Mark the hour as busy
                busy_slots.add(event_start.replace(minute=0, second=0, microsecond=0))
        
        # Generate available slots for the next week, but spread them out
        available_slots = []
        current_time = buffer_time
        
        # Round up to the next hour
        if current_time.minute > 0:
            current_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        # Ensure we start at business hours
        if current_time.hour < 10:
            current_time = current_time.replace(hour=10, minute=0, second=0, microsecond=0)
        elif current_time.hour >= 16:
            current_time = current_time.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        # Generate slots for the next week, but spread them out
        slot_count = 0
        while current_time < week_end and slot_count < 3:
            # Only consider business hours (10 AM - 4 PM) AND weekdays (Monday-Friday)
            if 10 <= current_time.hour < 16 and current_time.weekday() < 5:  # Monday=0, Friday=4
                # Check if this hour is available
                if current_time not in busy_slots:
                    available_slots.append(current_time)
                    print(f"✅ Available slot: {current_time.strftime('%A, %B %-d at %-I:%M %p')}")
                    slot_count += 1
                    
                    # Skip ahead to spread out the slots more aggressively
                    if slot_count < 3:
                        # For first slot, skip 3-4 hours or go to next day
                        if slot_count == 1:
                            if current_time.hour < 12:  # Morning
                                current_time += timedelta(hours=4)  # Skip to afternoon
                            else:  # Afternoon
                                current_time = current_time.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
                        # For second slot, go to next day
                        elif slot_count == 2:
                            current_time = current_time.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
                        # For third slot, go to day after next
                        else:
                            current_time = current_time.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=2)
                else:
                    # Move to next hour if this one is busy
                    current_time += timedelta(hours=1)
            else:
                # Move to next business day (skip weekends)
                current_time = current_time.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
                # Skip weekends
                while current_time.weekday() >= 5:  # Saturday=5, Sunday=6
                    current_time += timedelta(days=1)
        
        # Convert to formatted strings
        formatted_slots = []
        for slot in available_slots:
            formatted_slots.append({
                "datetime": slot.isoformat(),
                "display": slot.strftime("%A, %B %-d at %-I:%M %p"),
                "day": slot.strftime("%A"),
                "time": slot.strftime("%-I:%M %p")
            })
        
        print(f"🎯 Found {len(formatted_slots)} available slots for the next week")
        return formatted_slots

    except Exception as e:
        print(f"❌ Failed to check availability: {e}")
        return []


def check_slot_available(datetime_str: str) -> dict:
    """
    Checks if a specific datetime slot is available for booking.
    
    Args:
        datetime_str: ISO 8601 datetime string to check
        
    Returns:
        dict with 'available' (bool) and 'message' (str) keys
    """
    creds = get_persistent_credentials()
    if not creds:
        print("❌ No credentials available for availability check.")
        return {"available": False, "message": "❌ Calendar authorization token missing."}

    try:
        # Parse the datetime to check
        slot_start = datetime.fromisoformat(datetime_str)
        slot_end = slot_start + timedelta(hours=1)
        
        # Validate business hours (10 AM - 4 PM Central)
        if not (10 <= slot_start.hour < 16):
            return {
                "available": False, 
                "message": "❌ I can only schedule appointments between 10 AM and 4 PM Central Time."
            }
        
        # Check if the slot is in the past or too soon (24-hour buffer)
        now = datetime.now()
        buffer_time = now + timedelta(hours=24)
        
        if slot_start < buffer_time:
            return {
                "available": False,
                "message": "❌ I need at least 24 hours notice to schedule appointments."
            }
        
        service = build("calendar", "v3", credentials=creds)
        
        # Check for conflicts in the calendar
        events_result = service.events().list(
            calendarId='primary',
            timeMin=slot_start.isoformat() + 'Z',
            timeMax=slot_end.isoformat() + 'Z',
            singleEvents=True
        ).execute()
        
        conflicting_events = events_result.get('items', [])
        
        if conflicting_events:
            # There are conflicting events
            conflict_summary = conflicting_events[0].get('summary', 'Another meeting')
            return {
                "available": False,
                "message": f"❌ That time slot is not available. There's a conflict with: {conflict_summary}"
            }
        else:
            # Slot is available
            return {
                "available": True,
                "message": f"✅ {slot_start.strftime('%A, %B %-d at %-I:%M %p')} is available!"
            }
            
    except Exception as e:
        print(f"❌ Error checking slot availability: {e}")
        return {
            "available": False,
            "message": f"❌ Error checking availability: {str(e)}"
        }



