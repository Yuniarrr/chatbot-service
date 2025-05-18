import ftfy

from langchain.document_loaders import WebBaseLoader as BaseWebBaseLoader
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.utils.clean_text import clean_text


class CustomWebBaseLoader(BaseWebBaseLoader):
    def _scrape(
        self,
        url: str,
        parser: str | None = None,
        bs_kwargs: dict | None = None,
    ) -> BeautifulSoup:
        print("CustomWebBaseLoader")

        if parser is None:
            parser = "html.parser"

        self._check_parser(parser)

        # Fetch the raw HTML content
        html_doc = self.session.get(url, **self.requests_kwargs)
        if self.raise_for_status:
            html_doc.raise_for_status()

        if self.encoding is not None:
            html_doc.encoding = self.encoding
        elif self.autoset_encoding:
            html_doc.encoding = html_doc.apparent_encoding

        # Parse full HTML first
        soup = BeautifulSoup(html_doc.text, parser, **(bs_kwargs or {}))

        if "/dosen-staf/" in url:
            wrappers = soup.find_all("div", class_="wpb_wrapper")
            if len(wrappers) >= 2:
                html_str = "".join(str(w) for w in wrappers)
                return BeautifulSoup(html_str, parser)

        if "/himpunan-mahasiswa" in url:
            wrappers = soup.find_all("div", class_="wpb_text_column")
            html_str = "".join(str(w) for w in wrappers)
            return BeautifulSoup(html_str, parser)

        if "/admission" in url:
            wrappers = soup.find_all(
                "div", class_="wpb_text_column wpb_content_element"
            )
            html_str = "".join(str(w) for w in wrappers)
            return BeautifulSoup(html_str, parser)

        wrappers = soup.find_all("div", class_="wpb_text_column")
        if len(wrappers) >= 2:
            second_wrapper_html = str(wrappers[1])
            return BeautifulSoup(second_wrapper_html, parser)

        # 1. Try second div.wpb_wrapper
        wrappers = soup.find_all("div", class_="wpb_wrapper")
        if len(wrappers) >= 2:
            second_wrapper_html = str(wrappers[1])
            return BeautifulSoup(second_wrapper_html, parser)

        # 2. Try to find both div.title-news and div.content-news, combine their content
        title_news = soup.find("div", class_="title-news")
        content_news = soup.find("div", class_="content-news")

        if title_news and content_news:
            title_text = title_news.get_text(strip=True)
            content_text = content_news.get_text(separator="\n", strip=True)

            combined_text = f"Judul: {title_text}.\n\nIsi:\n{content_text}"

            # Wrap it in <div> so it remains valid HTML for the loader
            return BeautifulSoup(f"<div>{combined_text}</div>", parser)

        # 3. Fallback to just div.content-news if exists
        if content_news:
            return BeautifulSoup(str(content_news), parser)

        # If none found, return empty soup
        return BeautifulSoup(html_doc.text, parser, **(bs_kwargs or {}))

    def load(self):
        soup = self._scrape(self.web_path)

        if "/dosen-staf" in self.web_path:
            # Custom handling for dosen-staf page
            tables = soup.find_all("table")
            docs = []

            if "/daftar-dosen" in self.web_path:
                for table in tables:
                    rows = table.find_all("tr")
                    info = {}
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            key = cells[0].get_text(strip=True).replace("\xa0", " ")
                            value = (
                                cells[1].get_text(" ", strip=True).replace("\xa0", " ")
                            )
                            # Also fix text encoding issues if any
                            key = ftfy.fix_text(key)
                            value = ftfy.fix_text(value)
                            info[key] = value.strip(": ").strip()

                    name = info.get("Nama Dosen", "")
                    nip = info.get("NIP", "")
                    email = info.get("Email", "")
                    jabatan = info.get("Jabatan", "")
                    page_content = table.get_text("\n", strip=True)

                    doc = Document(
                        page_content=ftfy.fix_text(page_content),
                        metadata={
                            "name": name,
                            "nip": nip,
                            "email": email,
                            "jabatan": jabatan,
                            "source": self.web_path,
                        },
                    )
                    docs.append(doc)

            if "/daftar-staf-2" in self.web_path:
                containers = soup.find_all("div", class_="vc_tta-container")
                docs = []

                for container in containers:
                    # Find all panels in this staff block
                    panels = container.find_all("div", class_="vc_tta-panel")

                    for panel in panels:
                        heading = panel.find("span", class_="vc_tta-title-text")
                        if heading and "Data Kepegawaian" in heading.get_text():
                            table = panel.find("table")
                            if not table:
                                continue

                            rows = table.find_all("tr")
                            info = {}
                            for row in rows:
                                cells = row.find_all("td")
                                if len(cells) >= 2:
                                    key = cells[0].get_text(strip=True)
                                    value = (
                                        cells[1].get_text(" ", strip=True).lstrip(": ")
                                    )
                                    info[key] = value

                            name = info.get("Nama Staf", "")
                            npp = info.get("NPP", "")
                            fungsi = info.get("Fungsi", "")
                            email = info.get("Email", "")
                            text_content = table.get_text("\n", strip=True)

                            doc = Document(
                                page_content=ftfy.fix_text(text_content),
                                metadata={
                                    "name": clean_text(name),
                                    "npp": clean_text(npp),
                                    "fungsi": clean_text(fungsi),
                                    "email": clean_text(email),
                                    "source": self.web_path,
                                },
                            )
                            docs.append(doc)

            # if "/admission" in self.web_path:
            #     panels = soup.find_all("div", class_="vc_tta-panel")

            #     for panel in panels:
            #         tab_id = panel.get("id", "")
            #         title_tag = panel.find("h4", class_="vc_tta-panel-title")
            #         tab_title = title_tag.get_text(strip=True) if title_tag else tab_id

            #         body = panel.find("div", class_="vc_tta-panel-body")
            #         if body:
            #             text_content = body.get_text("\n", strip=True)
            #             cleaned_text = ftfy.fix_text(text_content)
            #             doc = Document(
            #                 page_content=cleaned_text,
            #                 metadata={
            #                     "tab": tab_title,
            #                     "tab_id": tab_id,
            #                     "source": self.web_path,
            #                 },
            #             )
            #             docs.append(doc)

            return docs

        # Default behavior (one document)
        text = soup.get_text(separator=" | ", strip=True)
        return [
            Document(
                page_content=ftfy.fix_text(text), metadata={"source": self.web_path}
            )
        ]
