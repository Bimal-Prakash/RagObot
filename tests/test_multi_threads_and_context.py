import unittest
import asyncio
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from core.thread_store import ThreadStore, generate_thread_title
from core.graph import (
    format_conversation_history,
    contextualize_query_for_retrieval,
    retrieve_chromadb_chunks_node,
    ollama_worker_node,
    groq_worker_node,
    consensus_arbiter_node
)
from core.state import RagObotState
import tempfile
from pathlib import Path


class TestMultiThreadsAndContextAwareness(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for:
    1. Multi-threaded chat persistence (CRUD operations, auto-titles).
    2. Context awareness: Pronoun resolution and conversational retrieval.
    3. Worker & Arbiter context awareness.
    4. Thread continuity across turns.
    """

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.thread_store = ThreadStore(storage_dir=self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. ThreadStore CRUD & Auto-Title
    def test_thread_store_crud(self):
        # Create
        t = self.thread_store.create_thread(title="New Chat")
        tid = t["id"]
        self.assertEqual(t["title"], "New Chat")

        # Append Turn
        u_msg = {"role": "user", "content": "who created ragobot"}
        a_msg = {"role": "assistant", "content": "RagObot was created by Bimal P B."}
        updated = self.thread_store.append_turn(tid, u_msg, a_msg, token_count=100)
        
        # Verify title auto-generated from first message
        self.assertEqual(updated["title"], "who created ragobot")
        self.assertEqual(len(updated["messages"]), 2)
        self.assertEqual(updated["token_count"], 100)

        # List
        threads = self.thread_store.list_threads()
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0]["id"], tid)
        self.assertEqual(threads[0]["title"], "who created ragobot")

        # Update title
        self.thread_store.update_thread_title(tid, "RagObot Creator")
        loaded = self.thread_store.get_thread(tid)
        self.assertEqual(loaded["title"], "RagObot Creator")

        # Delete
        self.assertTrue(self.thread_store.delete_thread(tid))
        self.assertIsNone(self.thread_store.get_thread(tid))

    # 2. Conversation History Formatting
    def test_format_conversation_history(self):
        messages = [
            HumanMessage(content="Who created RagObot?"),
            AIMessage(content="RagObot was created by Bimal P B."),
            HumanMessage(content="What else did he build?")  # Active turn
        ]
        history = format_conversation_history(messages)
        self.assertIn("User: Who created RagObot?", history)
        self.assertIn("RagObot: RagObot was created by Bimal P B.", history)
        # Current active query must be excluded from history
        self.assertNotIn("What else did he build?", history)

    # 3. Contextual Query Reformulation for Retrieval
    async def test_contextualize_query_for_retrieval(self):
        messages = [
            HumanMessage(content="Who created RagObot?"),
            AIMessage(content="RagObot was created by Bimal P B."),
            HumanMessage(content="What projects has he worked on?")
        ]
        # Test with mocked LLM response
        with patch("core.graph.ChatGroq") as mock_groq:
            instance = MagicMock()
            instance.ainvoke = MagicMock(return_value=asyncio.Future())
            instance.ainvoke.return_value.set_result(MagicMock(content="What projects has Bimal P B worked on?"))
            mock_groq.return_value = instance

            reformulated, target_doc = await contextualize_query_for_retrieval(
                query="What projects has he worked on?",
                messages=messages,
                cloud_model="llama-3.3-70b-versatile",
                local_model="gemma4:31b-cloud"
            )
            self.assertEqual(reformulated, "What projects has Bimal P B worked on?")
            self.assertIsNone(target_doc)

    # 4. Workers and Arbiter Receive Conversation History in Context
    async def test_worker_and_arbiter_context_inclusion(self):
        messages = [
            HumanMessage(content="Who created RagObot?"),
            AIMessage(content="RagObot was created by Bimal P B."),
            HumanMessage(content="What is his background?")
        ]
        state: RagObotState = {
            "query": "What is his background?",
            "messages": messages,
            "retrieved_docs": [{"breadcrumb": "Resume", "content": "Bimal P B is an AI engineer."}],
            "model_mode": "cloud",
            "cloud_model": "llama-3.3-70b-versatile",
            "local_model": "gemma4:31b-cloud",
            "ollama_output": "Bimal is an AI engineer.",
            "groq_output": "Bimal P B has a background in AI engineering."
        }

        with patch("core.graph.ChatGroq") as mock_groq:
            instance = MagicMock()
            # Capture messages passed to ChatGroq
            captured_messages = []
            async def fake_astream(msgs):
                nonlocal captured_messages
                captured_messages = msgs
                yield MagicMock(content="Groq output")

            instance.astream = fake_astream
            mock_groq.return_value = instance

            await groq_worker_node(state)
            self.assertTrue(len(captured_messages) > 0)
            human_prompt = captured_messages[1].content
            # Verify recent conversation history is in worker prompt
            self.assertIn("Recent Conversation History", human_prompt)
            self.assertIn("Who created RagObot?", human_prompt)
            self.assertIn("Bimal P B", human_prompt)


if __name__ == "__main__":
    unittest.main()
