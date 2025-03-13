import os
import shutil
import logging

from typing import BinaryIO, Tuple, Optional

from app.env import UPLOAD_DIR
from app.core.constants import ERROR_MESSAGES
from app.core.logger import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class UploaderService:
    def upload_to_local(self, contents: bytes, file_name: str) -> Tuple[bytes, str]:
        """Upload to /data/uploads"""
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file_name)
        with open(file_path, "wb") as f:
            f.write(contents)
        return contents, file_path

    def get_file_from_local(self, file_path: str) -> str:
        """Handles downloading of the file from local storage"""
        return file_path

    def delete_from_local(self, file_name: str) -> None:
        """Handles deletion of the file from local storage."""
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            log.warning("File {file_path} not found in local storage")

    def _delete_all_from_local(self) -> None:
        """Handles deletion of all files from local storage."""
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                except Exception as e:
                    log.error(f"Failed to delete {file_path}. Reason: {e}")
        else:
            log.warning(f"Directory {UPLOAD_DIR} not found in local storage")

    def upload_file(self, file: BinaryIO, filename: str) -> Tuple[bytes, str]:
        """Uploads a file to the local file system."""
        contents = file.read()
        if not contents:
            raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)
        contents, file_path = self.upload_to_local(contents, filename)

        return contents, file_path


uploader_service = UploaderService()
