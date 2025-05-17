import requests
import json
from bs4 import BeautifulSoup

base_urls = [
    {"link": "https://www.its.ac.id/it/id/category/berita/", "save_as": "berita"},
    {"link": "https://www.its.ac.id/it/id/mahasiswa/", "save_as": "mahasiswa"},
    {
        "link": "https://www.its.ac.id/it/id/riset-dan-kolaborasi/",
        "save_as": "riset-dan-kolaborasi",
    },
    {"link": "https://www.its.ac.id/it/id/akademik/", "save_as": "akademik"},
    {"link": "https://www.its.ac.id/it/id/fasilitas/", "save_as": "fasilitas"},
    {"link": "https://www.its.ac.id/it/id/tentang-kami/", "save_as": "tentang-kami"},
]


def scrape_n(start_url):
    visited_links = set()
    current_url = start_url

    while current_url:
        print(f"Scraping: {current_url}")
        try:
            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract news links
            post_divs = soup.find_all("div", class_="postgrid1")
            for div in post_divs:
                a = div.find("a")
                if a and a.get("href"):
                    visited_links.add(a["href"])

            # Get next page link
            nav = soup.find("div", class_="wp-pagenavi")
            if nav:
                next_link = nav.find("a", class_="nextpostslink")
                if next_link and next_link.get("href"):
                    current_url = next_link["href"]
                    continue
        except Exception as e:
            print(f"Error on {current_url}: {e}")

        # No next page
        break

    return list(visited_links)


# Loop through all base URLs
for item in base_urls:
    links = scrape_n(item["link"])
    filename = f'./scrape/{item["save_as"]}.json'

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(links)} links to {filename}")
