import requests
from bs4 import BeautifulSoup
import time

base_url = "https://www.shl.com/solutions/products/product-catalog/"
all_links = []

def fetch_page(url, retries=3, delay=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                print(f"⚠️ Non-200 status {response.status_code} on {url}")
        except requests.exceptions.RequestException as e:
            print(f"⛔ Error on {url} (attempt {attempt+1}): {e}")
        time.sleep(delay)
    return None

# Pagination loop
for i in range(0, 361, 12):
    url = f"{base_url}?start={i}&type=1"
    print(f"🔍 Scraping page: {url}")

    html = fetch_page(url)
    if html is None:
        print(f"❌ Skipping page {url} after 3 failed attempts.\n")
        continue

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table")

    page_links = []
    for table in tables:
        links = table.find_all("a", href=True)
        for link in links:
            text = link.text.strip()
            href = link['href'].strip()
            if href.startswith("/"):
                href = "https://www.shl.com" + href
            page_links.append((text, href))

    # Take only the last 12 links
    last_12 = page_links[-12:]
    all_links.extend(last_12)

    print(f"✅ Found {len(last_12)} links on this page.\n")
    time.sleep(1.5)

# Save to file
with open("shl_links.txt", "w", encoding="utf-8") as f:
    for text, href in all_links:
        f.write(f"{text} => {href}\n")

print(f"\n✅ All done! Scraped {len(all_links)} links across all pages.")
print("📁 Saved to: shl_links.txt")
