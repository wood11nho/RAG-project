# 🎓 RAG Course Assistant: Intelligent Retrieval for NLP

> **Project for Information Retrieval & Text Mining** > *Master’s in Natural Language Processing (Year 2)*

A specialized RAG (Retrieval-Augmented Generation) chatbot designed to answer student questions based **strictly** on course materials. Unlike standard LLMs (like ChatGPT), this assistant cites its sources (specific slides) and refuses to answer questions outside the syllabus, preventing hallucinations and academic dishonesty.

---

## 🏗️ Architecture

This project implements a **Hybrid Search** architecture to ensure maximum retrieval accuracy before generating answers:

1.  **Ingestion:** Converts `.pptx` (and legacy `.ppt`) files into clean text, preserving "Speaker Notes" which often contain critical context.
2.  **Retrieval Strategy (Ensemble):**
    * **Semantic Search (ChromaDB):** Uses `all-MiniLM-L6-v2` embeddings to find conceptually similar content.
    * **Keyword Search (BM25):** Uses sparse vector retrieval to catch specific terminology (e.g., "TF-IDF formula").
    * **Weighting:** **40% Keyword / 60% Semantic** weighted ensemble for balanced results.
3.  **Generation:** Passes retrieved context to **Llama 3.3 (70B Versatile)** via Groq for high-speed, grounded inference.

---

## 🚀 Key Features

* **📚 Grounded Answers:** Every claim is backed by a citation (e.g., `[Source: lecture1-intro.pptx]`).
* **🚫 Hallucination Guardrails:** If a topic (like "HITS Algorithm") is not in the slides, the bot explicitly refuses to answer.
* **⚡ Hybrid Retrieval:** Combines the precision of keyword matching with the understanding of vector search.
* **📊 Automated Benchmarking:** Includes a script to compare the RAG system against a "Pure LLM" and "Simple Search" automatically.

---

## 🛠️ Tech Stack

* **Orchestration:** LangChain
* **LLM Provider:** Groq (Llama-3.3-70b-versatile)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Sparse Retrieval:** RankBM25
* **Document Processing:** `python-pptx`, `unstructured`

---

## ⚙️ Setup & Installation

**Prerequisites:** Python 3.9+ and a [Groq API Key](https://console.groq.com/).

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd RAG-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have LibreOffice installed if you need to convert legacy .ppt files)*

3.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=gsk_your_key_here_...
    ```

4.  **Ingest Data:**
    Place your `.pptx` files in the `data/` folder and run:
    ```bash
    python ingest.py
    ```
    *This creates the `chroma_db/` folder and the `bm25_retriever.pkl` index.*

---

## 🏃‍♂️ Usage

### Interactive Chatbot
To start the assistant and ask questions in the terminal:
```bash
python main.py
```

### Run Evaluation
Generate the comparison report (RAG vs. LLM vs. Simple IR):
```bash
python benchmark.py
```
This produces `benchmark_results.csv` containing the 10 test queries.

## 📊 Evaluation & Results
Evaluated against three criteria (see `benchmark_results.csv` for full details):
* Existing topics: Answers correctly and cites slides.
* Missing topics: Correctly refuses to answer.
* General trivia: Refuses as out of domain.

Benchmark examples:
* In syllabus — "Explain TF-IDF": Pure LLM answers without citation; RAG answers and cites `lecture6-tfidf`.
* Missing topic — "Explain HITS Algorithm": Pure LLM hallucinates; RAG refuses ("Not in course materials").
* General trivia — "Capital of France": Pure LLM answers "Paris"; RAG refuses ("Not in course materials").

## 📜 License
This project is created for educational purposes within the "Information Retrieval & Text Mining" course.
