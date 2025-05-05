import os
from typing import List, Optional
import requests
import smtplib
import logging
import traceback

from datetime import datetime
from pydantic import BaseModel, EmailStr
from email.message import EmailMessage
from email.utils import make_msgid
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.core.logger import SRC_LOG_LEVELS
from app.env import GOOGLE_EMAIL, GOOGLE_PASSWORD, GOOGLE_CALENDAR_JSON
from app.models.feedbacks import FeedbackCreateModel, FeedbackType
from app.services.feedback import feedback_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


def get_current_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    :param city: The name of the city to get the weather for.
    :return: The current weather information or an error message.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "API key is not set in the environment variable 'OPENWEATHER_API_KEY'."
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # Optional: Use 'imperial' for Fahrenheit
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        data = response.json()
        if data.get("cod") != 200:
            return f"Error fetching weather data: {data.get('message')}"
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        return f"Weather in {city}: {temperature}°C"
    except requests.RequestException as e:
        return f"Error fetching weather data: {str(e)}"


class EmailInputSchema(BaseModel):
    email: EmailStr
    subject: str
    body: str


def send_email(email: str, subject: str, body: str) -> str:
    """
    Service pengiriman email.

    :param input: Dictionary yang berisi recipient_email, subject, dan body
    :return: Status pengiriman email
    """
    try:
        recipient_email = email

        # Set up email message
        message_data = EmailMessage()
        username = GOOGLE_EMAIL
        password = GOOGLE_PASSWORD
        message_data["Subject"] = subject
        message_data["From"] = username
        message_data["To"] = recipient_email

        # Add HTML content
        message_data.add_alternative(body, subtype="html")

        # Send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_server:
            smtp_server.login(username, password)
            smtp_server.send_message(message_data)

        return f"Berhasil mengirimkan email ke {recipient_email}"

    except smtplib.SMTPRecipientsRefused:
        # print("Recipient address rejected: not found")
        return "Alamat penerima tidak ditemukan"
    except smtplib.SMTPSenderRefused:
        # print("Sender address invalid")
        return "Alamat pengirim tidak valid"
    except smtplib.SMTPDataError:
        # print("The SMTP server refused to accept the message data.")
        return "Server SMTP menolak menerima data pesan"
    except Exception as error:
        log.error(f"Error: {error}")
        log.info(traceback.print_exc())
        return "Gagal mengirimkan email"


class CalendarInputSchema(BaseModel):
    summary: str
    start_datetime: datetime
    end_datetime: datetime
    attendees: List[str]
    description: Optional[str] = None
    location: Optional[str] = "Google Meet"


def add_to_calendar(
    summary: str,
    start_datetime: datetime,
    end_datetime: datetime,
    attendees: List[str],
    description: Optional[str] = None,
    location: Optional[str] = "Google Meet",
):
    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    TOKEN_PATH = os.path.join(BASE_DIR, "token.json")  # <-- token, NOT credentials
    CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")

    print("TOKEN_PATH")
    print(TOKEN_PATH)
    print("CREDENTIALS_PATH")
    print(CREDENTIALS_PATH)

    creds = None

    # Attempt to load existing token
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    # If no valid credentials, launch browser-based login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Requires user to log in once in browser
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_PATH, "w") as token_file:
                token_file.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary": "Test from CATI",
        "location": location,
        "description": description,
        "start": {
            "dateTime": start_datetime.isoformat(),
            "timeZone": "Asia/Jakarta",
        },
        "end": {
            "dateTime": end_datetime.isoformat(),
            "timeZone": "Asia/Jakarta",
        },
        "attendees": [{"email": email} for email in attendees],
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 24 * 60},
                {"method": "popup", "minutes": 10},
            ],
        },
    }

    event = service.events().insert(calendarId="primary", body=event).execute()
    return "Event created: %s" % (event.get("htmlLink"))


class FeedbackInputSchema(BaseModel):
    type: FeedbackType
    message: str
    sender: Optional[str] = None


async def add_to_feedback(
    type: FeedbackType, message: str, sender: Optional[str] = None
):
    try:
        _new_feedback = FeedbackCreateModel(
            **{"type": type, "message": message, "sender": sender}
        )

        await feedback_service.insert_new_file(_new_feedback)

        return (
            "Berhasil menambahkan feedback ke database. Terimakasih atas feedback nya"
        )
    except Exception as error:
        log.error(f"Error: {error}")
        log.info(traceback.print_exc())
        return "Terjadi kegagalan dalam menyimpan feedback ke database"
