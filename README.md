# AI-Powered Automobile Knowledge Assistant

## Problem Statement
Automobile manuals and knowledge bases are often extensive and difficult to navigate. Users struggle to quickly find specific specifications, maintenance procedures, or troubleshooting steps across multiple documents.

## Objective
Build a complete, working AI-Powered Automobile Knowledge Assistant using Retrieval-Augmented Generation (RAG). The assistant allows users to upload automobile documents (PDFs), ask conversational questions, and compare vehicles, providing grounded answers with genuine source citations (document name and page number).

## Features
- **PDF Document Processing**: Extracts text and performs automatic OCR on scanned pages using PyMuPDF and Tesseract.
- **Explainable RAG Workflow**: Semantic search using ChromaDB and Sentence Transformers to ground LLM answers, avoiding hallucinations.
- **Source Citations**: Displays exact document name and page number where the information was found. Includes expandable panels to view the exact retrieved text.
- **Multiple LLM Providers**: Supports Gemini (default), OpenAI, and Claude via a unified abstract factory.
- **Multi-Document Comparison**: Compare specifications and procedures from multiple uploaded manuals in a structured format.
- **Dynamic Dashboard**: Real-time insights into indexed documents, pages, chunks, and vectors.
- **Document Management**: Upload, view, re-index, and delete vehicle documents seamlessly.

## Technology Stack
- **Frontend:** Streamlit
- **Vector Database:** ChromaDB (Persistent local storage)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Document Processing:** PyMuPDF (fitz)
- **OCR:** pytesseract + Pillow
- **Chunking:** Langchain RecursiveCharacterTextSplitter
- **LLM APIs:** Google Gemini, OpenAI, Anthropic Claude
- **Language:** Python 3.11+

## Architecture & RAG Workflow
1. **Upload & Process:** Users upload PDFs. PyMuPDF extracts text; if text length is insufficient, Tesseract OCR extracts from the page image.
2. **Chunking:** Text is split using RecursiveCharacterTextSplitter, retaining metadata (document name, page number).
3. **Embeddings & Storage:** Text chunks are embedded via `SentenceTransformer` and stored persistently in `ChromaDB`.
4. **Retrieval:** User queries are embedded, and the top-K chunks are retrieved via Cosine Similarity.
5. **Generation & Explainability:** The LLM generates a grounded response from the context, and citations are extracted from the retrieved metadata to display alongside the answer.

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure Tesseract-OCR is installed on your system if you want to use the OCR features.*

4. **Environment Variables:**
   Copy `.env.example` to `.env` and fill in your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-2.0-flash

   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4o-mini

   ANTHROPIC_API_KEY=your_anthropic_api_key
   CLAUDE_MODEL=claude-3-haiku-20240307
   ```

## Running the Application
Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage Guide
- **Uploading Documents**: Go to `📄 Documents` and upload one or more automobile PDFs.
- **Searching**: Use `🔎 Search` for direct semantic lookup of chunk texts.
- **Asking Questions**: Navigate to `💬 Ask Assistant`. Select "All Documents" or filter by a specific manual.
- **Source Citations**: The AI's response will show the exact page and document. Expand the "View Retrieved Sources" box to verify.
- **Document Management**: Update or delete documents in the Documents tab. If uploading a file with the same name, it will automatically prompt or replace it.
- **LLM Switching**: Change the provider anytime in the `⚙ Settings` page.

## Troubleshooting
- **No relevant information found?** Ensure the document was successfully processed and the text isn't entirely images without OCR availability.
- **Missing API Key error?** Ensure `.env` is configured correctly and restart the app.

## Future Enhancements
- Support for complex tables and diagrams extraction.
- Conversational state summarization for extremely long chats.
- Hybrid search combining BM25 and vector search.
