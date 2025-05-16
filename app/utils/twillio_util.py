import httpx
import base64
import os
import uuid

from app.env import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
from app.services.uploader import uploader_service


async def download_twilio_media(url: str):
    async with httpx.AsyncClient(
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), follow_redirects=True
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        extension = content_type.split("/")[-1] if "/" in content_type else "bin"
        filename = f"{uuid.uuid4()}.{extension}"

        contents = response.content  # raw bytes
        _, file_path = uploader_service.upload_to_local(contents, filename)

        with open(file_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("utf-8")

        return {
            "file_data": file_data,
            "filename": filename,
            "content_type": content_type,
            "filename": filename,
            "final_url": str(response.url),
        }
