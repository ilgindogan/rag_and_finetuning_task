import pandas as pd

# Read CSV
docs_df = pd.read_csv("./data/documents.csv")
qaps_df = pd.read_csv("./data/qaps.csv")
matched_df = pd.read_csv("./data/matched_gutenberg_books.csv")

# Find document id if include test set.
test_doc_ids = qaps_df[qaps_df['set'] == 'test']['document_id'].unique()

# filter gutenberg
gutenberg_docs = docs_df[(docs_df['kind'] == 'gutenberg') & (docs_df['document_id'].isin(test_doc_ids))]

# match the wiki_title from matched_gutenberg_books.csv, and find candidates books.
matched_titles = matched_df['title'].tolist()
candidates = gutenberg_docs[gutenberg_docs['wiki_title'].isin(matched_titles)]

# create df for suitable books given this rule : test and gutenberg.
selection_df = candidates[['document_id', 'wiki_title', 'story_url']]
selection_df = selection_df.drop_duplicates().reset_index(drop=True)

print(selection_df)

# define book id as " -0.txt"
selected_doc_id = "eaba6723befdddc5c5177c5c1bf5262d850c2a99"
selected_title = "The Castle of Otranto"

# filter test set for selected book
filtered_qa = qaps_df[(qaps_df['document_id'] == selected_doc_id) & (qaps_df['set'] == 'test')]
print(f"Finding QA count:{len(filtered_qa)}")

# Create Q/A data
qa_data = filtered_qa[['question', 'answer1', 'answer2']]
qa_data.to_csv(f"./data/{selected_title.lower().replace(' ', '_')}_qa_test.csv", index=False)
qa_data.to_json(f"./data/{selected_title.lower().replace(' ', '_')}_qa_test.json", orient='records', lines=True)

print("All process completed")
