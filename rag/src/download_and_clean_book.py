import requests
import re
import unicodedata

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
replacements = {
        "—": "-", "–": "-",
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "…": "...", "¶": "",
    }
for src, tgt in replacements.items():
    cleaned_text = cleaned_text.replace(src, tgt)

# Unicode normalize 
cleaned_text = unicodedata.normalize("NFKC", cleaned_text)

# Clean chapter title
cleaned_text = re.sub(r'\bCHAPTER\s+[IVXLCDM]+\b', '', cleaned_text, flags=re.IGNORECASE)

# Other titles
cleaned_text = re.sub(r'\bVOLUME\s+[IVXLCDM]+\b', '', cleaned_text, flags=re.IGNORECASE)
cleaned_text = re.sub(r'^\*{3,}$', '', cleaned_text, flags=re.MULTILINE)

lines = cleaned_text.splitlines()
cleaned_lines = [line.strip() for line in lines if len(line.strip().split()) > 2]
cleaned_text = "\n".join(cleaned_lines)


# Save
with open(f"./data/{selected_title.lower().replace(' ', '_')}_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("Book is cleaned and saved")
