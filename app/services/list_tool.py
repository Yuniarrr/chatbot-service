import os
import requests


def get_current_weather(self, city: str) -> str:
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
