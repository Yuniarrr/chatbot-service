import logging
import ftfy

# from langchain_community.document_loaders.parsers.images import TesseractBlobParser
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document

from app.core.logger import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class Loader:
    def __init__(self, engine: str = "", **kwargs):
        self.engine = engine
        self.kwargs = kwargs

    def load(
        self, file_name: str, file_content_type: str, file_path: str
    ) -> list[Document]:
        try:
            print(file_name, file_content_type, file_path)
            loader = self._get_loader(file_name, file_content_type, file_path)
            print(loader)
            print("sebelum docs load")
            docs = loader.load()

            print("pass docs load")

            return [
                Document(
                    page_content=ftfy.fix_text(doc.page_content), metadata=doc.metadata
                )
                for doc in docs
            ]
        except Exception as e:
            print(f"Error load file: {e}")
            log.error(f"Error load file: {e}")

    def _get_loader(self, file_name: str, file_content_type: str, file_path: str):
        file_ext = file_name.split(".")[-1].lower()

        if file_ext == "pdf":
            loader = PyPDFLoader(file_path, extract_images=True)
            # loader = PyMuPDFLoader(
            #     file_path,
            #     extract_images=self.kwargs.get("PDF_EXTRACT_IMAGES"),
            #     images_parser=TesseractBlobParser(langs=["eng"]),
            #     extract_tables="markdown",
            #     mode="page",
            # )
        elif file_ext == "csv":
            loader = CSVLoader(file_path)
        elif file_ext in ["htm", "html"]:
            loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
        elif file_ext == "md":
            loader = TextLoader(file_path, autodetect_encoding=True)
        elif file_content_type == "application/epub+zip":
            loader = UnstructuredEPubLoader(file_path)
        elif (
            file_content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or file_ext == "docx"
        ):
            loader = Docx2txtLoader(file_path)
        elif file_content_type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ] or file_ext in ["xls", "xlsx"]:
            loader = UnstructuredExcelLoader(file_path)
        elif file_content_type in [
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ] or file_ext in ["ppt", "pptx"]:
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_ext == "msg":
            loader = OutlookMessageLoader(file_path)
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)

        return loader
