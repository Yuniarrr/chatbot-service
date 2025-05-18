import json
import time
import requests

API_BASE_URL = "http://localhost:8080/api/v1/dev/file"
COLLECTION_NAME = "fasilitas"
FILE_LOCATION = "./scrape/fasilitas.json"
STATUS_CHECK_INTERVAL = 5  # seconds
MAX_ATTEMPTS = 10  # to avoid infinite loops
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InN0cmluZyIsImlkIjoiZTFhZGJkNWMtMDY5ZS00MGEzLTliYzMtM2U2ZDg2ODQ3NjY3Iiwicm9sZSI6IkFETUlOSVNUUkFUT1IiLCJleHAiOjE3NDc1NjQ5NDd9.9eEzMDXeAvprOKmPvXQNhNgcyZ_btn79exlx8MuUlYc"

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/x-www-form-urlencoded",
}


def submit_file(url):
    payload = {"url": url, "collection_name": COLLECTION_NAME}
    try:
        response = requests.post(API_BASE_URL, data=payload, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        return data["data"]["id"]
    except requests.RequestException as e:
        print(f"Error submitting file: {e}")
        return None


def check_status(file_id):
    try:
        url = f"{API_BASE_URL}/{file_id}"
        for attempt in range(MAX_ATTEMPTS):
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            status = response.json()["data"]["status"]
            print(f"Status check {attempt + 1}: {status}")
            if status in ["SUCCESS", "FAILED"]:
                return status
            time.sleep(STATUS_CHECK_INTERVAL)
        print("Max attempts reached without final status.")
        return "UNKNOWN"
    except requests.RequestException as e:
        print(f"Error checking status: {e}")
        return "ERROR"


def main():
    with open(FILE_LOCATION) as f:
        urls = json.load(f)

    for url in urls:
        print(f"\nSubmitting: {url}")
        file_id = submit_file(url)
        if not file_id:
            print("Skipping due to submission error.")
            continue

        status = check_status(file_id)
        if status == "FAILED":
            print("Processing failed. Stopping further processing.")
            break
        elif status == "SUCCESS":
            print("Processing succeeded. Continuing to next URL.")
        else:
            print(f"Unexpected status '{status}'. Stopping.")
            break


if __name__ == "__main__":
    main()
