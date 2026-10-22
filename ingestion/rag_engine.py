import os
import shutil
from pathlib import Path
from typing import Any, Optional
import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import config

class RAGDocumentChunk:
    def __init__(self, content: str, metadata: dict[str, Any], source_file: str):
        self.content = content
        self.metadata = metadata
        self.source_file = source_file

    def get_breadcrumb(self) -> str:
        """Constructs hierarchical breadcrumb from markdown headers."""
        headers = []
        for level in ["Header 1", "Header 2", "Header 3", "Header 4"]:
            if level in self.metadata:
                headers.append(str(self.metadata[level]))
        path_str = " > ".join(headers) if headers else "General"
        return f"[{self.source_file} :: {path_str}]"

    def to_context_str(self) -> str:
        return f"{self.get_breadcrumb()}\n{self.content}"

class RAGEngine:
    """Production ChromaDB vector pipeline powered by Nomic embeddings (nomic-embed-text)."""
    
    COLLECTION_NAME = "ragobot_knowledge"

    def __init__(self, data_dir: Optional[Path] = None, chroma_dir: Optional[Path] = None):
        self.data_dir = data_dir or config.DATA_DIR
        self.chroma_dir = chroma_dir or config.CHROMA_DIR
        self.chunks: list[RAGDocumentChunk] = []
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        
        # Configure Markdown Splitter
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

    def _get_client(self) -> chromadb.PersistentClient:
        """Initializes persistent ChromaDB storage."""
        if self._chroma_client is None:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        return self._chroma_client

    def _get_collection(self):
        """Retrieves or creates the cosine-similarity ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _get_embeddings(self) -> Optional[OllamaEmbeddings]:
        """Initializes Nomic embeddings via Ollama."""
        if self._embeddings is None:
            try:
                self._embeddings = OllamaEmbeddings(
                    model=config.EMBEDDING_MODEL,
                    base_url=config.OLLAMA_BASE_URL
                )
            except Exception as e:
                print(f"[!] Warning: Unable to initialize Nomic embeddings: {e}")
                self._embeddings = None
        return self._embeddings

    def ingest_documents(self) -> int:
        """
        Parses Markdown, TXT, and PDF files in data/, embeds them using
        Nomic (nomic-embed-text), and indexes them into persistent ChromaDB.
        """
        self.chunks.clear()
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return 0
            
        # 1. Ingest Markdown files
        md_files = list(self.data_dir.glob("*.md"))
        for file_path in md_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                md_splits = self.md_splitter.split_text(raw_text)
                final_splits = self.sub_splitter.split_documents(md_splits)
                for doc in final_splits:
                    chunk = RAGDocumentChunk(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        source_file=file_path.name
                    )
                    self.chunks.append(chunk)
            except Exception as e:
                print(f"[!] Warning: Error reading {file_path.name}: {e}")

        # 2. Ingest Plain Text files
        txt_files = list(self.data_dir.glob("*.txt"))
        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                splits = self.sub_splitter.split_text(raw_text)
                for idx, split_text in enumerate(splits):
                    chunk = RAGDocumentChunk(
                        content=split_text,
                        metadata={"Header 1": f"Section {idx + 1}"},
                        source_file=file_path.name
                    )
                    self.chunks.append(chunk)
            except Exception as e:
                print(f"[!] Warning: Error reading {file_path.name}: {e}")

        # 3. Ingest PDF files
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if pdf_files:
            try:
                from pypdf import PdfReader
                for file_path in pdf_files:
                    try:
                        reader = PdfReader(str(file_path))
                        for page_num, page in enumerate(reader.pages, start=1):
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                splits = self.sub_splitter.split_text(page_text)
                                for s in splits:
                                    chunk = RAGDocumentChunk(
                                        content=s,
                                        metadata={"Header 1": f"Page {page_num}"},
                                        source_file=file_path.name
                                    )
                                    self.chunks.append(chunk)
                    except Exception as e:
                        print(f"[!] Warning: Error processing PDF {file_path.name}: {e}")
            except ImportError:
                print("[!] Warning: pypdf not installed for PDF ingestion.")
                
        # 4. Populate ChromaDB Vector Store with Nomic Embeddings
        if self.chunks:
            client = self._get_client()
            try:
                client.delete_collection(self.COLLECTION_NAME)
            except Exception:
                pass
            self._collection = client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            embedder = self._get_embeddings()
            ids = [f"chunk_{i}_{c.source_file}" for i, c in enumerate(self.chunks)]
            documents = [c.content for c in self.chunks]
            metadatas = [
                {
                    "source": c.source_file,
                    "breadcrumb": c.get_breadcrumb()
                }
                for c in self.chunks
            ]
            
            if embedder:
                try:
                    embeddings = embedder.embed_documents(documents)
                    self._collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                except Exception as e:
                    print(f"[!] Error creating Nomic embeddings in ChromaDB: {e}")
                    # Fallback to ChromaDB internal text addition
                    self._collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
            else:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
        return len(self.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K_CHUNKS,
        target_doc: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Retrieves top-k relevant chunks from ChromaDB using Nomic query embeddings.
        Supports document-focused retrieval when target_doc is specified.
        """
        collection = self._get_collection()
        total_docs = collection.count()
        if total_docs == 0:
            return []
            
        embedder = self._get_embeddings()
        retrieved = []
        seen_keys = set()

        # 1. Target Document Focus: If a specific document is targeted (e.g. just uploaded)
        if target_doc:
            clean_target = Path(target_doc).name
            try:
                # Query within the target document
                if embedder:
                    query_embedding = embedder.embed_query(query)
                    target_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(top_k, total_docs),
                        where={"source": clean_target},
                        include=["documents", "metadatas", "distances"]
                    )
                    t_docs = target_results.get("documents", [[]])[0]
                    t_metas = target_results.get("metadatas", [[]])[0]
                    t_dists = target_results.get("distances", [[]])[0]
                    for doc_text, meta, dist in zip(t_docs, t_metas, t_dists):
                        score = round(max(0.0, 1.0 - float(dist)), 4)
                        breadcrumb = meta.get("breadcrumb", "General")
                        source = meta.get("source", clean_target)
                        formatted = f"{breadcrumb}\n{doc_text}"
                        if formatted not in seen_keys:
                            retrieved.append({
                                "content": doc_text,
                                "breadcrumb": breadcrumb,
                                "source": source,
                                "score": score,
                                "formatted": formatted
                            })
                            seen_keys.add(formatted)

                # Also fetch direct document chunks in natural order if it's an overview/structural query
                direct_chunks = collection.get(
                    where={"source": clean_target},
                    include=["documents", "metadatas"]
                )
                d_docs = direct_chunks.get("documents", [])
                d_metas = direct_chunks.get("metadatas", [])
                for doc_text, meta in zip(d_docs, d_metas):
                    breadcrumb = meta.get("breadcrumb", "General")
                    source = meta.get("source", clean_target)
                    formatted = f"{breadcrumb}\n{doc_text}"
                    if formatted not in seen_keys and len(retrieved) < top_k:
                        retrieved.append({
                            "content": doc_text,
                            "breadcrumb": breadcrumb,
                            "source": source,
                            "score": 0.85,
                            "formatted": formatted
                        })
                        seen_keys.add(formatted)
            except Exception as e:
                print(f"[!] Target doc retrieval note ({clean_target}): {e}")

        # 2. General Semantic Retrieval across all indexed documentation
        remaining_slots = max(0, top_k - len(retrieved))
        if remaining_slots > 0 or not retrieved:
            try:
                n_results = min(top_k, total_docs)
                if embedder:
                    query_embedding = embedder.embed_query(query)
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                else:
                    results = collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]
                
                for doc_text, meta, dist in zip(docs, metas, dists):
                    score = round(max(0.0, 1.0 - float(dist)), 4)
                    breadcrumb = meta.get("breadcrumb", "General")
                    source = meta.get("source", "unknown")
                    formatted = f"{breadcrumb}\n{doc_text}"
                    if formatted not in seen_keys:
                        retrieved.append({
                            "content": doc_text,
                            "breadcrumb": breadcrumb,
                            "source": source,
                            "score": score,
                            "formatted": formatted
                        })
                        seen_keys.add(formatted)
            except Exception as e:
                print(f"[!] ChromaDB retrieval error: {e}")

        return retrieved[:max(top_k, len(retrieved))] if target_doc else retrieved[:top_k]

# Global singleton
rag_engine = RAGEngine()
