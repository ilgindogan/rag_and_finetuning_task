import requests
import re
# define book title
selected_title = "The Castle of Otranto"

# Gutenberg URL
book_url = "https://www.gutenberg.org/files/696/696-0.txt"
response = requests.get(book_url)
text = response.text
print(text)

# Gutenberg header/footer cleaning
pattern = r"\*\*\* START OF.*?\*\*\*(.*?)\*\*\* END OF.*?\*\*\*"
match = re.search(pattern, text, re.DOTALL)

if match:
    cleaned_text = match.group(1).strip()
else:
    raise ValueError("No find start end blocks")
# clean text extra space
cleaned_text = '\n'.join([line.strip() for line in cleaned_text.splitlines() if line.strip()])

# Save
with open(f"./data/{selected_title.lower().replace(' ', '_')}_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("Book is cleaned and saved")
