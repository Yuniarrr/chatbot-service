import logging
from typing import Union
import ftfy
import pytesseract
import re
import aiohttp
import requests
import bs4

# from langchain_community.document_loaders.parsers.images import TesseractBlobParser
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from PIL import Image
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
    WebBaseLoader,
)
from langchain_core.documents import Document
from pdf2image import convert_from_path

from app.core.logger import SRC_LOG_LEVELS
from app.retrieval.custom_loader import CustomWebBaseLoader

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
            print(f"Loading file: {file_name}")
            file_ext = file_name.split(".")[-1].lower()
            print(file_name, file_content_type, file_path)
            loader = self._get_loader(file_name, file_content_type, file_path)

            if loader is None and file_ext in ["jpg", "jpeg", "png"]:
                print("Handling image file with OCR")
                return self._ocr_image(file_path)

            print(loader)
            print("sebelum docs load")
            docs = loader.load()

            print("pass docs load")

            fixed_docs = [
                Document(
                    page_content=ftfy.fix_text(doc.page_content), metadata=doc.metadata
                )
                for doc in docs
            ]

            # Cek apakah semua hasilnya kosong atau whitespace
            if all(doc.page_content.strip() == "" for doc in fixed_docs):
                print("Semua page_content kosong. Fallback ke OCR.")
                return self._ocr_pdf(file_path)

            return fixed_docs
        except Exception as e:
            print(f"Fallback to OCR due to error: {e}")
            log.warning(f"OCR fallback for {file_name} because: {e}")
            return self._ocr_pdf(file_path)

    def load_url(self, url: str) -> list[Document]:
        try:
            print(f"Loading URL: {url}")
            loader = self._get_loader_url(url)
            docs = loader.load()

            if not docs:
                raise ValueError("Tidak ada dokumen yang dihasilkan dari loader.")

            print("pass docs load url")
            print(docs)

            fixed_docs = []
            for doc in docs:
                cleaned_content = self._clean_page_content(doc.page_content)
                structured_content = self._structure_paragraphs(cleaned_content)
                truncated = self._remove_intro_until_bantuan(structured_content)

                fixed_doc = Document(
                    page_content=ftfy.fix_text(truncated),
                    metadata=doc.metadata,
                )
                fixed_docs.append(fixed_doc)

            print("pass fixed docs")
            print(fixed_docs)
            return fixed_docs
        except Exception as e:
            print(f"Error load: {e}")
            log.warning(f"Error load {url} because: {e}")
            return []

    def _sanitize_pdf_header(self, file_path):
        print("Memeriksa header PDF...")
        with open(file_path, "rb") as f:
            content = f.read()
        index = content.find(b"%PDF")
        if index > 0:
            print("PDF header ditemukan bukan di awal file, disesuaikan...")
            with open(file_path, "wb") as f:
                f.write(content[index:])

    def _get_loader(self, file_name: str, file_content_type: str, file_path: str):
        file_ext = file_name.split(".")[-1].lower()

        if file_ext == "pdf":
            self._sanitize_pdf_header(file_path)
            loader = PyPDFLoader(file_path, extract_images=True)
        elif file_ext == "csv":
            loader = CSVLoader(file_path)
        elif file_ext in ["htm", "html"]:
            loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
        elif file_ext == "md":
            loader = TextLoader(file_path, autodetect_encoding=True)
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
            loader = UnstructuredPowerPointLoader(
                file_path,
                mode="elements",
                strategy="fast",
            )
        elif file_ext == "msg":
            loader = OutlookMessageLoader(file_path)
        elif file_ext in ["jpg", "jpeg", "png"]:
            return None
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)

        return loader

    def _get_loader_url(self, url: str):
        print("WebBaseLoader")
        loader = CustomWebBaseLoader(
            web_path=url,
            bs_get_text_kwargs={"separator": " | ", "strip": True},
        )

        return loader

    def has_classes(tag):
        return tag.name and all(
            cls in tag.get("class", []) for cls in ["title-news", "content-news"]
        )

    def _bs4_extractor(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

    def _metadata_extractor(
        self,
        raw_html: str,
        url: str,
        response: Union[requests.Response, aiohttp.ClientResponse],
    ) -> dict:
        content_type = getattr(response, "headers").get("Content-Type", "")
        return {"source": url, "content_type": content_type}

    def _get_base_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _ocr_pdf(self, file_path: str) -> list[Document]:
        try:
            images = convert_from_path(
                file_path, poppler_path=r"C:\poppler-24.08.0\Library\bin"
            )
            text = ""

            for idx, image in enumerate(images):
                try:
                    page_text = pytesseract.image_to_string(image)
                    text += page_text + "\n"
                except Exception as page_error:
                    print(f"OCR gagal pada halaman {idx + 1}: {page_error}")
                    log.warning(f"OCR gagal pada halaman {idx + 1}: {page_error}")
                    continue  # skip halaman ini

            if not text.strip():
                log.error("OCR selesai tapi hasil kosong.")
                return []

            temp_txt_path = file_path + ".ocr.txt"
            with open(temp_txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            loader = TextLoader(temp_txt_path, autodetect_encoding=True)
            docs = loader.load()

            return [
                Document(
                    page_content=ftfy.fix_text(doc.page_content),
                    metadata=doc.metadata,
                )
                for doc in docs
            ]
        except Exception as ocr_error:
            log.error(f"OCR processing failed: {ocr_error}")
            return []

    def _ocr_image(self, file_path: str) -> list[Document]:
        try:
            text = pytesseract.image_to_string(Image.open(file_path))
            return [
                Document(
                    page_content=ftfy.fix_text(text.strip()),
                    metadata={"source": file_path},
                )
            ]
        except Exception as e:
            log.error(f"OCR failed for image {file_path}: {e}")
            return []

    def _clean_page_content(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s*\|\s*", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _structure_paragraphs(self, text: str) -> str:
        sentences = re.split(r"(?<=[.?!])\s+", text)
        paragraphs = "\n\n".join(
            [" ".join(sentences[i : i + 3]) for i in range(0, len(sentences), 3)]
        )
        return paragraphs

    def _remove_intro_until_bantuan(self, text: str) -> str:
        index = text.find(": Bantuan")
        if index != -1:
            return text[index + len("Bantuan") :].lstrip()
        return text


class CustomITSLoader(WebBaseLoader):
    def _scrape(self, url: str, **kwargs) -> str:
        print(f"[DEBUG] Scraping: {url}")
        res = self.session.get(url)
        soup = BeautifulSoup(res.text, "lxml")

        # Ambil judul dari vc_column-inner ke-2
        judul_divs = soup.find_all("div", class_="vc_column-inner")
        judul = (
            judul_divs[1].find("h3").get_text(strip=True) if len(judul_divs) > 1 else ""
        )

        # Ambil konten dari wpb_wrapper ke-2
        wrapper_divs = soup.find_all("div", class_="wpb_wrapper")
        konten = (
            wrapper_divs[1].get_text(separator="\n", strip=True)
            if len(wrapper_divs) > 1
            else ""
        )

        return judul + "\n\n" + konten
