import os
import time
import requests

# === CONFIGURATION ===
API_BASE_URL = "http://localhost:8080/api/v1/dev/file"
COLLECTION_NAME = "pengumuman"
FILE_DIR = r"E:\woww\dataset\pengumuman"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InN0cmluZyIsImlkIjoiZTFhZGJkNWMtMDY5ZS00MGEzLTliYzMtM2U2ZDg2ODQ3NjY3Iiwicm9sZSI6IkFETUlOSVNUUkFUT1IiLCJleHAiOjE3NDc1NjQ5NDd9.9eEzMDXeAvprOKmPvXQNhNgcyZ_btn79exlx8MuUlYc"

# === TIMING ===
STATUS_CHECK_INTERVAL = 5  # seconds between status checks
MAX_ATTEMPTS = 10  # max retries for status polling

# === HEADERS ===
HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
}


def submit_file(filepath):
    """Send the file to the API using multipart/form-data."""
    try:
        with open(filepath, "rb") as file_obj:
            files = {"file": (os.path.basename(filepath), file_obj)}
            data = {"collection_name": COLLECTION_NAME}
            response = requests.post(
                API_BASE_URL, headers=HEADERS, files=files, data=data
            )
            response.raise_for_status()
            return response.json()["data"]["id"]
    except Exception as e:
        print(f"[❌] Upload failed for {filepath}: {e}")
        return None


def check_status(file_id):
    """Poll the file status until it's SUCCESS or FAILED."""
    try:
        status_url = f"{API_BASE_URL}/{file_id}"
        for attempt in range(MAX_ATTEMPTS):
            response = requests.get(status_url, headers=HEADERS)
            response.raise_for_status()
            status = response.json()["data"]["status"]
            print(f"[🔍] Status attempt {attempt + 1}: {status}")
            if status in ["SUCCESS", "FAILED"]:
                return status
            time.sleep(STATUS_CHECK_INTERVAL)
        return "UNKNOWN"
    except Exception as e:
        print(f"[❌] Error checking status for file ID {file_id}: {e}")
        return "ERROR"


def main():
    files = sorted(
        [f for f in os.listdir(FILE_DIR) if os.path.isfile(os.path.join(FILE_DIR, f))]
    )

    print(f"Found {len(files)} files. Starting upload...\n")

    for index, filename in enumerate(files, start=1):
        full_path = os.path.join(FILE_DIR, filename)
        print(f"[📤] ({index}/{len(files)}) Uploading: {filename}")

        file_id = submit_file(full_path)
        if not file_id:
            print("[⚠️] Skipping file due to upload error.\n")
            continue

        status = check_status(file_id)
        if status == "FAILED":
            print("[❌] File processing failed. Stopping further uploads.\n")
            break
        elif status == "SUCCESS":
            print("[✅] File processed successfully.\n")
        else:
            print(f"[⚠️] Unexpected status '{status}'. Stopping.\n")
            break


if __name__ == "__main__":
    main()
