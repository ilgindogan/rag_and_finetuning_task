# RAG and Finetuning Task – *The Castle of Otranto*

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline with **hierarchical chunking** to answer questions from *The Castle of Otranto*. The pipeline is evaluated against a simple generation baseline (no retrieval) using NarrativeQA-style QA pairs.

Key tasks include:
- Selecting and cleaning a book (Gutenberg).
- Hierarchical and semantic chunking.
- Embedding and indexing with Qdrant.
- Retrieval, prompting, and generation.
- Evaluation using ROUGE,  BLEU and cosine similarity.

---

## Folder Structure

```
rag-and-finetuning-task/
│
├── rag/
│   ├── data/                                           # Book text, QA pairs, chunked data
│       ├-- otranto_hierarchical_chunks.jsonl
│       ├-- otranto_semantic_chunks.jsonl
│       ├-- the_castle_of_otranto_cleaned.txt
│       ├-- the_castle_of_otranto_qa_test.csv
│       ├-- the_castle_of_otranto_qa_test.json
│       ├-- documents.csv
│       ├-- ...
│   ├── qdrant_data/                # Persisted Qdrant index
│   ├── results/                    # Model outputs as jsonl format
│   └── src/                        # Main pipeline scripts
│       ├-- chunk_and_embed.py
│       ├-- evaluate.py
│       ├-- run_rag.py
│       |-- no_rag_run.py
│       |-- qdrant_indexing.py
│       |-- cosine_chunk_embedding.py
│       |-- download_and_clean_book.py
│       |-- generate_QA_list_selected_book.py
│       |-- find_suitable_book_from_gutenberg.py
|       
│   └── test_scripts/              # QA evaluation & analysis tools
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

##  Setup Instructions

###  Python Environment

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```
- Also, I used conda environments for specific python version.
```bash
conda create -n ilgin python=3.10.18
pip install -r requirements.txt
```

###  Models & Embeddings

Make sure you have downloaded or cached:
- Sentence-transformer model (e.g., `intfloat/e5-base-v2`)
- HuggingFace LLM (`google/gemma-3-1b-it`)


###  Qdrant Setup (Local)

```bash
pip install qdrant-client
# The project uses local Qdrant with disk persistence in /qdrant_data
```
- In addition, you can set docker for local accessing.

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```
---

##  How to Run

### 1. Clean and Prepare the Book

```bash
python src/download_and_clean_book.py
```

### 2. Generate QA Pairs

```bash
python src/generate_QA_list_selected_book.py
```

### 3. Chunk and Embed (Hierarchical)

```bash
python src/chunk_and_embed.py
```

### 4. Index into Qdrant

```bash
python src/qdrant_indexing.py
```

### 5. Run RAG Pipeline

```bash
python src/run_rag.py
```

### 6. Baseline Run (No RAG)

```bash
python src/no_rag_run.py
```

### 7. Evaluate

```bash
python src/evaluate.py
```

---

##  Chunking Strategy

- **Parent chunks**: ~138 tokens
- **Child chunks**: ~686 tokens
- Each child retains a reference to its parent to allow *context stitching* during retrieval.

This chunks is storing as hierarchical strategy in results folder.
- `otranto_hierarchical_chunks.jsonl`

I also experimented with **semantic chunking**, stored in: This method includes semantic + hierarchical method

- `otranto_semantic_chunks.jsonl`

---

##  Embedding Models

I tested multiple sentence embeddings:
- `all-MiniLM-L6-v2`
- `bge-small-en`
- `e5-base-v2`
- Optionally, Instructor embeddings

Comparative analysis is found in:
- `test_scripts/comparing_embedding_models.py`

Also, this script analyze different embedding model result in rag pipeline with llm.
- `test_scripts/rag_with_llm.py`

Other analysis and script is related with relevance chunking and cosine-rouge analysis of rag.
- `test_scripts/chunk_relevance_cosine.py`
- `test_scripts/chunk_relevance_rouge.py`
- `test_scripts/rouge_analiz.py`
- `test_scripts/cosine_analysis.py`
---

##  Retrieval Strategy

- Top-k child chunk retrieval (`k=5`,`k=10`)
- Linked parent context is added to the prompt
- Cosine similarity used for semantic relevance
- Each strategy is calculated based time consuming.

---

##  Evaluation Results

| Approach             | ROUGE-L | BLEU              |
|----------------------|---------|-------------------|
| No-RAG (LLM only)    | 0.0624  | 0.0131            |
| RAG (Hierarchical)   | 0.1909  | 0.0158            |

Also I calculated the cosine similarity score for both evaluation.

| Approach             | Cosine Similarity|
|----------------------|------------------|
| No-RAG (LLM only)    | 0.7444           | 
| RAG (Hierarchical)   | 0.7978           |

More analysis is available in:
- `results/rag_answers.jsonl`
- `results/no_rag_answers.jsonl`

---

##  Resources

- **Book**: *The Castle of Otranto* (Gutenberg)
- **QA Data**: Custom or NarrativeQA-style
- **Vector DB**: Local Qdrant

---

##  Report

See: [`report.md`](./report.md)

---

##  Conclusion

This project demonstrates that **hierarchical RAG** improves answer quality in long documents. Future improvements may include:
- Advanced reranking (e.g., Cohere ReRank)
- Long-context LLMs
- Better semantic chunking strategies + hierarchical chunking 

---

##  Author

Ilgın | July 2025
