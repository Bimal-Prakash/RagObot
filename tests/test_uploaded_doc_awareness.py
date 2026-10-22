import unittest
import asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from core.graph import (
    is_document_referencing_query,
    contextualize_query_for_retrieval,
    retrieve_chromadb_chunks_node
)
from core.state import RagObotState
from core.thread_store import ThreadStore
from ingestion.rag_engine import rag_engine
import config


class TestUploadedDocAwareness(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying that when a document is uploaded,
    queries like 'what are the things inside this' correctly identify
    and retrieve the just uploaded document.
    """

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.thread_store = ThreadStore(storage_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_document_referencing_detection(self):
        """Verify queries referring to 'this', 'this document', 'inside this', etc."""
        doc_queries = [
            "what are the things inside this",
            "what is inside this",
            "what are the things in this",
            "what is in this document",
            "what does this document say",
            "summarize this",
            "tell me about this doc",
            "explain this file",
            "what are the contents of this"
        ]
        for q in doc_queries:
            self.assertTrue(
                is_document_referencing_query(q),
                f"Expected '{q}' to be recognized as document-referencing"
            )

        non_doc_queries = [
            "who created ragobot",
            "what is the capital of France",
            "how do local and cloud models collaborate",
            "what projects has Bimal worked on"
        ]
        for q in non_doc_queries:
            self.assertFalse(
                is_document_referencing_query(q),
                f"Expected '{q}' to NOT be recognized as document-referencing"
            )

    async def test_contextualize_query_with_uploaded_doc(self):
        """Verify contextualizer resolves 'this' to the uploaded document."""
        # 1. Deterministic fast resolution
        rephrased, target = await contextualize_query_for_retrieval(
            query="what are the things inside this",
            messages=[],
            cloud_model="llama-3.3-70b-versatile",
            local_model="gemma4:31b-cloud",
            uploaded_doc="Diet plan.txt"
        )
        self.assertEqual(target, "Diet plan.txt")
        self.assertIn("Diet plan.txt", rephrased)

        # 2. Detailed question using LLM rephrasing
        with patch("core.graph.ChatGroq") as mock_groq:
            instance = MagicMock()
            instance.ainvoke = MagicMock(return_value=asyncio.Future())
            instance.ainvoke.return_value.set_result(
                MagicMock(content="What does Diet plan.txt say about breakfast and protein intake?")
            )
            mock_groq.return_value = instance

            rephrased_llm, target_llm = await contextualize_query_for_retrieval(
                query="what does it say about breakfast and protein?",
                messages=[],
                cloud_model="llama-3.3-70b-versatile",
                local_model="gemma4:31b-cloud",
                uploaded_doc="Diet plan.txt"
            )
            self.assertEqual(target_llm, "Diet plan.txt")
            self.assertEqual(rephrased_llm, "What does Diet plan.txt say about breakfast and protein intake?")

    def test_thread_store_uploaded_doc_persistence(self):
        """Verify thread store records and retrieves uploaded document association."""
        t = self.thread_store.create_thread(title="Diet Plan Discussion")
        tid = t["id"]
        self.assertIsNone(self.thread_store.get_thread_uploaded_doc(tid))

        # Associate document
        self.thread_store.set_thread_uploaded_doc(tid, "Diet plan.txt")
        self.assertEqual(self.thread_store.get_thread_uploaded_doc(tid), "Diet plan.txt")

        # Reload from disk
        reloaded = self.thread_store.get_thread(tid)
        self.assertEqual(reloaded["last_uploaded_doc"], "Diet plan.txt")

    def test_rag_engine_retrieve_target_doc(self):
        """Verify rag_engine prioritizes chunks from target_doc."""
        # Ingest documents ensuring Diet plan.txt is present
        data_file = config.DATA_DIR / "Diet plan.txt"
        source_file = config.BASE_DIR / "Diet plan.txt"
        if source_file.exists() and not data_file.exists():
            shutil.copy(source_file, data_file)
        rag_engine.ingest_documents()

        results = rag_engine.retrieve(
            query="what are the things inside this",
            top_k=3,
            target_doc="Diet plan.txt"
        )
        self.assertTrue(len(results) > 0)
        # First chunk MUST be from Diet plan.txt
        self.assertEqual(results[0]["source"], "Diet plan.txt")


if __name__ == "__main__":
    unittest.main()
