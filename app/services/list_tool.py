import os
from typing import Dict, List, Optional
import requests
import smtplib
import logging
import traceback
import asyncio

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from email.message import EmailMessage
from email.utils import make_msgid
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from app.core.logger import SRC_LOG_LEVELS
from app.env import GOOGLE_EMAIL, GOOGLE_PASSWORD, GOOGLE_CALENDAR_JSON
from app.models.feedbacks import FeedbackCreateModel, FeedbackType
from app.models.opportunities import OpportunitiesCreateModel, OpportunityType
from app.services.feedback import feedback_service
from app.services.opportunity import opportunity_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class QueryInput(BaseModel):
    query: str


class AskConsent(BaseModel):
    """Memastikan apakah pengguna ingin menyimpan data."""

    action: str


def ask_consent_tool(action: str):
    return f"Sebelum saya menyimpan data, apakah Anda yakin ingin melanjutkan dengan {action}? (ya/tidak)"


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
    type: FeedbackType = Field(
        ..., description="Jenis feedback, misalnya NEGATIVE atau POSITIVE"
    )
    message: str = Field(..., description="Isi dari feedback yang ingin diberikan")
    sender: Optional[str] = Field(
        None,
        description="Nama atau identitas pengirim feedback. Kosongkan jika ingin anonim.",
    )


async def add_to_feedback(
    type: FeedbackType, message: str, sender: Optional[str] = "anon"
):
    try:
        _new_feedback = FeedbackCreateModel(
            **{"type": type, "message": message, "sender": sender}
        )

        await feedback_service.insert_new_feedback(_new_feedback)

        return (
            "Berhasil menambahkan feedback ke database. Terimakasih atas feedback nya"
        )
    except Exception as error:
        log.error(f"Error: {error}")
        log.info(traceback.print_exc())
        return "Terjadi kegagalan dalam menyimpan feedback ke database"


class OpportunityInputSchema(BaseModel):
    type: Optional[OpportunityType] = Field(
        ...,
        description="Jenis progran atau opportunity, misalnya BEASISWA, MAGANG, LOMBA, SERTIFIKASI, SEMINAR",
    )
    title: Optional[str] = Field(..., description="Nama program atau opportunity")
    skip: Optional[int] = Field(
        default=0,
        description="Parameter ini menentukan jumlah data yang harus dilewati sebelum mulai mengambil data. Misalnya, jika skip diatur ke 10, maka sistem akan melewatkan 10 data pertama dan mulai mengambil data setelahnya. Ini berguna saat Anda ingin mengakses data dari halaman yang lebih dalam atau memulai pengambilan data dari titik tertentu.",
    )
    limit: Optional[int] = Field(
        default=10,
        description="Parameter ini menentukan jumlah maksimum data yang akan diambil. Misalnya, jika limit diatur ke 10, maka hanya 10 data pertama (setelah melewati skip jika ada) yang akan diambil dan dikembalikan.",
    )


async def get_opportunity(
    type: Optional[OpportunityType] = None,
    title: Optional[str] = None,
    skip: Optional[int] = 0,
    limit: Optional[int] = 10,
):
    try:
        opportunities = await opportunity_service.get_opportunity_by_filter(
            type=type,
            title=title,
            skip=skip,
            limit=limit,
        )

        # Jika data ditemukan, return hasilnya
        if opportunities:
            return opportunities
        else:
            return "Tidak ada data peluang yang ditemukan."
    except Exception as error:
        log.error(f"Error: {error}")
        log.info(traceback.print_exc())
        return "Terjadi kegagalan dalam mengambil data dari database opportunity"


class AddNewOpportunityInputSchema(BaseModel):
    title: str = Field(description="Nama program atau opportunity")
    description: Optional[str] = Field(
        None, description="Deskripsi lebih lengkap mengenai program atau opportunity"
    )
    organizer: Optional[str] = Field(
        None, description="Penyelenggara program atau opportunity"
    )
    type: OpportunityType = Field(
        None,
        description="Jenis progran atau opportunity, misalnya BEASISWA, MAGANG, LOMBA, SERTIFIKASI, SEMINAR",
    )
    start_date: Optional[str] = Field(
        None,
        description="Tanggal dimulainya program atau opportunity dengan format YYYY-MM-DD",
    )
    end_date: Optional[str] = Field(
        None,
        description="Tanggal dimulainya program atau opportunity dengan format YYYY-MM-DD",
    )
    link: Optional[str] = Field(
        None,
        description="URL lainnya yang mungkin perlu ditambahkan",
    )
    image_url: Optional[str] = Field(
        None,
        description="URL gambar atau poster kegiatan",
    )
    sender: str = Field(
        None,
        description="Nama atau identitas pengirim opportunity. Kosongkan jika ingin anonim.",
    )


async def add_new_opportunity(
    title: str,
    type: OpportunityType,
    description: Optional[str] = None,
    organizer: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    link: Optional[str] = None,
    image_url: Optional[str] = None,
    sender: Optional[str] = None,
):
    try:
        _new_opportunity = OpportunitiesCreateModel(
            **{
                "title": title,
                "description": description,
                "organizer": organizer,
                "type": type,
                "start_date": start_date,
                "end_date": end_date,
                "link": link,
                "image_url": image_url,
                "uploader": sender,
            }
        )
        await opportunity_service.insert_new_opportunity(form_data=_new_opportunity)
        return f"Berhasil menambahkan {type.name.lower()} ke database. Terimakasih atas penambahan datanya."
    except Exception as error:
        log.error(f"Error: {error}")
        log.info(traceback.print_exc())
        return "Terjadi kegagalan dalam menyimpan data opportunity ke database"


class CollectionSelectorInput(BaseModel):
    query: str
    collections: List[str]


class CollectionChoice(BaseModel):
    chosen_collection: str = Field(
        description="Nama koleksi yang paling relevan dengan query"
    )


async def select_collection(query: str, collections: List[Dict[str, str]]) -> str:
    """Gunakan LLM untuk memilih nama koleksi paling relevan dari daftar berdasarkan query."""
    try:
        model = init_chat_model("gpt-4o", model_provider="openai")
        parser = JsonOutputParser(pydantic_object=CollectionChoice)

        prompt = PromptTemplate(
            template=(
                "Berikut adalah query dari pengguna:\n"
                "{query}\n\n"
                "Dan ini daftar koleksi:\n"
                "{collections}\n\n"
                "{format_instructions}"
            ),
            input_variables=["query", "collections"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | model | parser

        collections_str = "\n".join(
            f"- {col['name']}: {col['description']}" for col in collections
        )

        response_dict = await chain.ainvoke(
            {"query": query, "collections": collections_str}
        )

        # response_dict adalah dict, akses dengan key
        return response_dict["chosen_collection"]
    except Exception as e:
        print(f"Parsing error: {e}")
        return "Tidak dapat menentukan koleksi relevan"
