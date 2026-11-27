import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

# Load variables from .env into environment
load_dotenv()


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))


class GraphRAG:
    def __init__(
        self,
        llm_model: str | None = None,
        embedding_model: str | None = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

        self.client = genai.Client(api_key=api_key,http_options=types.HttpOptions(api_version="v1beta"),)
        self.llm_model = llm_model or os.getenv("LLM_MODEL")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL")

        self.graph = nx.DiGraph()
        self.chunks: Dict[str, Chunk] = {}
        self.entity_embeddings: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 1. Document Loading & Chunking
    # ------------------------------------------------------------------
    @staticmethod
    def load_documents_from_folder(docs_dir: str) -> List[Tuple[str, str]]:
        """Reads .txt, .md, and .pdf files recursively from docs_dir."""
        folder = Path(docs_dir)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(
                f"Docs directory not found: {folder.resolve()}"
            )

        documents: List[Tuple[str, str]] = []
        supported_exts = {".txt", ".md", ".pdf"}

        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in supported_exts:
                continue

            text = ""
            if file_path.suffix.lower() in {".txt", ".md"}:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            elif file_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(file_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages)

            text = text.strip()
            if text:
                documents.append((str(file_path.relative_to(folder)), text))

        return documents

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 100
    ) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks

    def get_embedding(self, text: str) -> np.ndarray:
        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text.replace("\n", " "),
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

    # ------------------------------------------------------------------
    # 2. Entity & Relationship Extraction
    # ------------------------------------------------------------------
    def extract_graph_elements(self, text: str) -> Dict:
        system_prompt = (
            "You are a knowledge graph extraction engine. Extract key entities "
            "and their relationships from the provided text. Return ONLY valid JSON "
            "matching this schema:\n"
            "{\n"
            '  "entities": [{"name": "Entity Name", "type": "Type", "description": "Brief description"}],\n'
            '  "relationships": [{"source": "Entity A", "target": "Entity B", "relation": "RELATION_TYPE"}]\n'
            "}"
        )

        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, TypeError):
            return {"entities": [], "relationships": []}

    # ------------------------------------------------------------------
    # 3. Build Knowledge Graph & Vector Index
    # ------------------------------------------------------------------
    def index_folder(self, docs_dir: str) -> None:
        documents = self.load_documents_from_folder(docs_dir)
        if not documents:
            raise ValueError(f"No readable .txt, .md, or .pdf files found in {docs_dir}")

        print(f"Loaded {len(documents)} document(s) from '{docs_dir}'.")
        chunk_counter = 0

        for source_name, doc_text in documents:
            print(f"  Indexing: {source_name}")
            for raw_chunk in self.chunk_text(doc_text):
                chunk_id = f"chunk_{chunk_counter}"
                chunk_counter += 1

                chunk_emb = self.get_embedding(raw_chunk)
                self.chunks[chunk_id] = Chunk(
                    chunk_id=chunk_id,
                    source=source_name,
                    text=raw_chunk,
                    embedding=chunk_emb,
                )

                extracted = self.extract_graph_elements(raw_chunk)

                for ent in extracted.get("entities", []):
                    name = ent.get("name", "").strip().lower()
                    if not name:
                        continue

                    if self.graph.has_node(name):
                        self.graph.nodes[name]["chunk_ids"].add(chunk_id)
                        if ent.get("description"):
                            self.graph.nodes[name]["descriptions"].append(
                                ent["description"]
                            )
                    else:
                        self.graph.add_node(
                            name,
                            display_name=ent["name"].strip(),
                            entity_type=ent.get("type", "Unknown"),
                            descriptions=[ent.get("description", "")],
                            chunk_ids={chunk_id},
                        )

                for rel in extracted.get("relationships", []):
                    src = rel.get("source", "").strip().lower()
                    tgt = rel.get("target", "").strip().lower()
                    relation = rel.get("relation", "RELATED_TO").strip()

                    if not src or not tgt:
                        continue

                    for node_key, raw_name in [(src, rel["source"]), (tgt, rel["target"])]:
                        if not self.graph.has_node(node_key):
                            self.graph.add_node(
                                node_key,
                                display_name=raw_name.strip(),
                                entity_type="Unknown",
                                descriptions=[],
                                chunk_ids={chunk_id},
                            )

                    self.graph.add_edge(
                        src, tgt, relation=relation, chunk_id=chunk_id
                    )

        for node, data in self.graph.nodes(data=True):
            desc_text = f"{data['display_name']} ({data['entity_type']}): " + " ".join(
                data["descriptions"]
            )
            self.entity_embeddings[node] = self.get_embedding(desc_text)

    # ------------------------------------------------------------------
    # 4. Hybrid Graph + Vector Retrieval
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom != 0 else 0.0

    def retrieve(
        self, query: str, top_k_entities: int = 3, hops: int = 2, top_k_chunks: int = 2
    ) -> Tuple[List[str], List[Chunk]]:
        query_emb = self.get_embedding(query)

        entity_scores = [
            (node, self._cosine_similarity(query_emb, emb))
            for node, emb in self.entity_embeddings.items()
        ]
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        seed_entities = [node for node, _ in entity_scores[:top_k_entities]]

        undirected_graph = self.graph.to_undirected()
        visited_nodes: Set[str] = set()
        for seed in seed_entities:
            if seed in undirected_graph:
                ego_nodes = nx.ego_graph(undirected_graph, seed, radius=hops).nodes()
                visited_nodes.update(ego_nodes)

        graph_facts: List[str] = []
        relevant_chunk_ids: Set[str] = set()

        for node in visited_nodes:
            relevant_chunk_ids.update(self.graph.nodes[node].get("chunk_ids", set()))

        for u, v, data in self.graph.edges(data=True):
            if u in visited_nodes and v in visited_nodes:
                u_name = self.graph.nodes[u]["display_name"]
                v_name = self.graph.nodes[v]["display_name"]
                graph_facts.append(f"({u_name}) -[{data['relation']}]-> ({v_name})")
                if "chunk_id" in data:
                    relevant_chunk_ids.add(data["chunk_id"])

        chunk_scores = [
            (cid, self._cosine_similarity(query_emb, chunk.embedding))
            for cid, chunk in self.chunks.items()
        ]
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        for cid, _ in chunk_scores[:top_k_chunks]:
            relevant_chunk_ids.add(cid)

        retrieved_chunks = [self.chunks[cid] for cid in relevant_chunk_ids]
        return graph_facts, retrieved_chunks

    # ------------------------------------------------------------------
    # 5. Answer Generation
    # ------------------------------------------------------------------
    def query_with_context(
        self, question: str, top_k_entities: int = 3, hops: int = 2
    ) -> Tuple[str, List[str], List[Chunk]]:
        graph_facts, retrieved_chunks = self.retrieve(
            question, top_k_entities=top_k_entities, hops=hops
        )

        context = (
            "### Knowledge Graph Relationships:\n"
            + ("\n".join(f"- {fact}" for fact in graph_facts) or "None")
            + "\n\n### Source Text Excerpts:\n"
            + "\n\n".join(
                f"[{i+1}] (Source: {c.source})\n{c.text}"
                for i, c in enumerate(retrieved_chunks)
            )
        )

        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=f"Context:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Answer the user's question using ONLY the provided Knowledge Graph "
                    "relationships and source text excerpts. Cite the source filenames "
                    "and specific connections when performing multi-hop reasoning."
                ),
                temperature=0.2,
            ),
        )
        return response.text, graph_facts, retrieved_chunks

    def query(self, question: str, top_k_entities: int = 3, hops: int = 2) -> str:
        answer, _, _ = self.query_with_context(
            question, top_k_entities=top_k_entities, hops=hops
        )
        return answer


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    docs_dir = os.getenv("DOCS_DIR", "./docs")

    rag = GraphRAG()
    rag.index_folder(docs_dir)

    print(
        f"\nGraph ready: {rag.graph.number_of_nodes()} nodes, "
        f"{rag.graph.number_of_edges()} edges.\n"
        "Type your question (or 'exit' to quit):\n"
    )

    while True:
        question = input("Ask > ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        answer = rag.query(question, top_k_entities=3, hops=2)
        print(f"\n{answer}\n" + "-" * 60 + "\n")