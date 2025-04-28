import os
import requests
import smtplib
import logging
import traceback

from pydantic import BaseModel, EmailStr
from email.message import EmailMessage
from email.utils import make_msgid

from app.core.logger import SRC_LOG_LEVELS
from app.env import GOOGLE_EMAIL, GOOGLE_PASSWORD

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
