import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import GraphRAG

rag: GraphRAG | None = None
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = GraphRAG()
    try:
        rag.index_folder(DOCS_DIR)
    except Exception as e:
        print(f"Startup indexing warning: {e}")
    yield


app = FastAPI(title="GraphRAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k_entities: int = 3
    hops: int = 2


@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/stats")
def get_stats():
    if not rag:
        return {"nodes": 0, "edges": 0, "chunks": 0}
    return {
        "nodes": rag.graph.number_of_nodes(),
        "edges": rag.graph.number_of_edges(),
        "chunks": len(rag.chunks),
    }


@app.post("/api/reindex")
def reindex_docs():
    global rag
    try:
        new_rag = GraphRAG()
        new_rag.index_folder(DOCS_DIR)
        rag = new_rag
        return {
            "status": "ok",
            "nodes": rag.graph.number_of_nodes(),
            "edges": rag.graph.number_of_edges(),
            "chunks": len(rag.chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def query_rag(req: QueryRequest):
    if not rag or rag.graph.number_of_nodes() == 0:
        raise HTTPException(
            status_code=400,
            detail="Knowledge graph is empty. Add files to docs/ and click Re-index.",
        )

    answer, graph_facts, retrieved_chunks = rag.query_with_context(
        req.question, top_k_entities=req.top_k_entities, hops=req.hops
    )

    return {
        "answer": answer,
        "graph_facts": graph_facts,
        "sources": [
            {"source": c.source, "text": c.text} for c in retrieved_chunks
        ],
    }