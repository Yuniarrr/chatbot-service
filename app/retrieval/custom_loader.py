from langchain.document_loaders import WebBaseLoader as BaseWebBaseLoader
from bs4 import BeautifulSoup


class CustomWebBaseLoader(BaseWebBaseLoader):
    def _scrape(
        self,
        url: str,
        parser: str | None = None,
        bs_kwargs: dict | None = None,
    ) -> BeautifulSoup:
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
