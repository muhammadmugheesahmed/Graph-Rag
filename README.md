# GraphRAG Explorer

GraphRAG Explorer is a local document question-answering application that combines:

- **Large-language-model entity and relationship extraction**
- **A NetworkX knowledge graph**
- **Vector similarity search for documents and entities**
- **Google Gemini embeddings and text generation**
- **A FastAPI backend and a browser-based chat interface**

Add plain text, Markdown, or PDF documents to `docs/`, run the indexer, and ask direct or multi-hop questions through the web UI or REST API.

## Features

- Recursive indexing of `.txt`, `.md`, and `.pdf` files
- Automatic word-based chunking with overlap
- Entity and relationship extraction using Gemini
- Directed knowledge graph stored with NetworkX
- Hybrid retrieval using graph traversal and cosine similarity
- Configurable seed entities and graph hop count
- Source excerpts and retrieved graph triples visible in the UI
- CLI and web/API interfaces
- Browser endpoint for re-indexing documents

## Architecture

```text
Documents (.txt, .md, .pdf)
        |
        v
Chunking + Gemini Embeddings
        |
        +----------------------+
        |                      |
        v                      v
Knowledge Graph         Chunk Vector Index
  (NetworkX)            (NumPy embeddings)
        |                      |
        +----------+-----------+
                   |
                   v
       Hybrid Graph + Vector Retrieval
                   |
                   v
       Gemini Answer with Source Context
```

The indexing flow is implemented in [`rag.py`](rag.py):

1. Documents are loaded recursively from `docs/`.
2. Text is split into 500-word chunks with 100-word overlap.
3. Gemini creates an embedding for every chunk.
4. Gemini extracts entities and relationships from each chunk.
5. Entities become graph nodes and relationships become directed edges.
6. Gemini generates embeddings for entity descriptions.

At query time, the application finds the most relevant entity seeds, traverses the graph, retrieves nearby triples and chunks, and sends that context to Gemini for answer generation.

## Technology Stack

- **Python 3.11+**
- **FastAPI** and **Uvicorn** for the HTTP API
- **Google Gen AI SDK** for Gemini embeddings and generation
- **NetworkX** for knowledge-graph storage and traversal
- **NumPy** for embedding and cosine-similarity operations
- **python-dotenv** for environment variables
- **PyPDF** for PDF text extraction
- **Vanilla HTML, CSS, and JavaScript** for the browser UI

## Repository Layout

```text
.
|-- docs/
|   |-- clients.txt
|   |-- projects.txt
|   `-- supply_chain.txt
|-- index.html                  # Web chat interface
|-- rag.py                      # GraphRAG indexing and retrieval engine
|-- server.py                   # FastAPI application and REST API
|-- test.py                     # Gemini model connectivity example
|-- requirements.txt            # Python dependencies
|-- sample_questions_to_ask.txt # Example GraphRAG queries
|-- .env                        # Local API configuration (do not commit)
`-- README.md
```

## Prerequisites

- Python 3.11 or newer
- A Google Gemini API key
- Internet access for Gemini API calls

## Installation

From Windows PowerShell, open the project directory:

```powershell
cd "C:\Users\mughe\OneDrive\Desktop\Agentic & Graph RAG\graph rag"
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow local scripts for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
DOCS_DIR=./docs
```

Environment variables:

| Variable | Required | Description |
|---|---:|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `LLM_MODEL` | Yes* | Model used for extraction and answer generation |
| `EMBEDDING_MODEL` | Yes* | Model used for chunk and entity embeddings |
| `DOCS_DIR` | No | Document directory; defaults to `./docs` |

\* If not supplied, model names can also be passed directly to `GraphRAG(...)`; otherwise initialization does not set a model fallback.

Never commit `.env` or expose your API key in source control. Add `.env` to `.gitignore` and rotate the key immediately if it has been exposed.

## Adding Documents

Place supported files in `docs/` or any subdirectory below it:

```text
docs/
|-- policies/
|   `-- remote-work.md
|-- reports/
|   `-- annual-report.pdf
`-- clients.txt
```

Supported formats:

- `.txt`
- `.md`
- `.pdf`

After adding or editing documents, use the **Re-index docs/** button in the UI or call `POST /api/reindex`.

## Running the Web Application

Start the FastAPI server:

```powershell
python -m uvicorn server:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The server indexes `DOCS_DIR` during startup. The UI can re-index documents without restarting the process.

## Running the CLI

Start the command-line interface:

```powershell
python rag.py
```

Example session:

```text
Graph ready: 20 nodes, 32 edges.
Type your question (or 'exit' to quit):

Ask > What is Project Skyfall?
```

Type `exit` or `quit` to stop the session.

## Using the API

### Get index statistics

```http
GET /api/stats
```

Example response:

```json
{
  "nodes": 18,
  "edges": 27,
  "chunks": 6
}
```

### Re-index documents

```http
POST /api/reindex
```

Example response:

```json
{
  "status": "ok",
  "nodes": 18,
  "edges": 27,
  "chunks": 6
}
```

### Ask a question

```http
POST /api/query
Content-Type: application/json
```

Request:

```json
{
  "question": "Why is the October 2026 launch of Project Skyfall at risk?",
  "top_k_entities": 3,
  "hops": 2
}
```

Response:

```json
{
  "answer": "Generated answer...",
  "graph_facts": [
    "(Project Skyfall) -[DEPENDS_ON]-> (Falcon-X9)"
  ],
  "sources": [
    {
      "source": "projects.txt",
      "text": "Retrieved document text..."
    }
  ]
}
```

Example PowerShell request:

```powershell
$body = @{
    question = "Who is the lead architect of Project Skyfall?"
    top_k_entities = 3
    hops = 2
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Retrieval Settings

The UI exposes two retrieval controls:

| Setting | Default | Effect |
|---|---:|---|
| Top-K seed entities | 3 | Embedding-matches used as the starting nodes |
| Graph hops | 2 | Maximum graph distance around each seed node |

Increase the hop count for more complex cross-document questions. Higher values can retrieve more relationships and produce more context but may also introduce noise.

## Example Questions

The repository includes [sample questions](sample_questions_to_ask.txt). Typical examples include:

```text
Who is the lead architect of Project Skyfall?
Why is the October 2026 launch of Project Skyfall at risk?
How could Kenji Sato's announcement affect Nexus Dynamics' Robotics Division?
```

For the final example, use a graph hop count of `3`.

## Testing the Gemini Connection

[`test.py`](test.py) prints the Gemini models that support `generateContent`, then sends a simple prompt:

```powershell
python test.py
```

Use this script when diagnosing API key, SDK, model, or network problems.

## Troubleshooting

### `GEMINI_API_KEY is missing`

- Confirm that `.env` exists in the project root.
- Confirm that `GEMINI_API_KEY` is set.
- Restart the terminal or server after changing `.env`.

### `Knowledge graph is empty`

- Confirm that supported files exist in `DOCS_DIR`.
- Check server startup output for indexing errors.
- Add documents and click **Re-index docs/**.

### No supported documents found

Only `.txt`, `.md`, and `.pdf` files are indexed. Confirm that files are not empty and use UTF-8 encoding for text and Markdown files.

### PDF text is missing

Some PDFs contain only scanned images and require OCR before the text can be extracted. PyPDF does not perform OCR.

### API quota or model errors

Confirm that:

- The selected model is available in your Gemini project.
- Your API key has access to the embedding and generation models.
- Your account has sufficient quota.
- `LLM_MODEL` and `EMBEDDING_MODEL` match available model names.

### Activation policy error on Windows

Run PowerShell as your normal user and apply an execution-policy change only to the current terminal process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Limitations

- The graph and vector indexes are held in memory and rebuilt when the server starts or re-indexes.
- There is no persistent database or disk-backed vector index.
- Extraction quality depends on the selected Gemini model and source text quality.
- Chunk overlap can repeat context in the index.
- Answers are grounded in retrieved context but can still contain model-generated errors.
- The API has no authentication by default and should not be exposed publicly without adding access controls.
- Re-indexing is a blocking synchronous operation.

## Security

- Keep the Gemini API key out of source control.
- Do not commit `.env`.
- Add authentication and rate limiting before Internet deployment.
- Use HTTPS and restrict CORS or network access for production use.
- Review generated answers before using them for decisions involving sensitive or regulated data.

## Contributing

1. Create a feature branch.
2. Keep indexing, retrieval, API, and UI changes scoped and tested.
3. Add or update sample documents and questions when changing retrieval behavior.
4. Do not commit API keys, virtual environments, generated indexes, or personal documents.
5. Submit a pull request with setup notes and validation steps.

## License

No license is currently specified in this repository. Add a `LICENSE` file if the project will be distributed publicly.