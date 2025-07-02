import pandas as pd
import requests
from urllib.parse import quote
from time import sleep

# NarrativeQA documents download
docs_file = "./data/documents.csv"

docs_df = pd.read_csv(docs_file)

# filter for gutenberg
gutenberg_docs = docs_df[docs_df['kind'] == 'gutenberg']

# list of book
titles = gutenberg_docs["wiki_title"].dropna().unique().tolist()

# finding the books with gutenberg api
def search_gutenberg(title):
    api_url = f"https://gutendex.com/books/?search={quote(title)}"
    try:
        resp = requests.get(api_url, timeout=10)
        data = resp.json()
        results = data.get("results", [])
        if results:
            first = results[0]
            return {
                "title": title,
                "gutenberg_id": first['id'],
                "matched_title": first['title'],
                "download_url": next(
                    (url for fmt, url in first['formats'].items()
                     if fmt.endswith('text/plain; charset=utf-8') or fmt.endswith('text/plain')), None
                )
            }
    except Exception as e:
        print(f"Error: {title} search fail for - {e}")
        return {"title": title, "error": str(e)}
    return {"title": title, "gutenberg_id": None}

# applied for loop and filter to all books
results = []
for i, t in enumerate(titles):
    print(f"[{i+1}/{len(titles)}] Searching: {t}")
    match = search_gutenberg(t)
    if match.get("gutenberg_id") and match.get("download_url"):
        results.append(match)
    sleep(1)
# create dataframe and save csv
books_df = pd.DataFrame(results)
books_df = books_df[['title', 'matched_title', 'gutenberg_id', 'download_url']]
books_df.to_csv("matched_gutenberg_books.csv", index=False)