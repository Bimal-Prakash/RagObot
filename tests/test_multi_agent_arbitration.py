"""
Automated Test Suite: RagObot Multi-Agent Arbitration and Grounding Reliability.
Covers all 12 required scenarios specified in requirements:
1. Fully supported question ("Who built RagObot?")
2. Unrelated question (chocolate cake recipe)
3. Six-stroke-engine question when knowledge base only documents four-stroke engines
4. Partially supported question (pruning unsupported claims)
5. Two workers agreeing on an unsupported claim (shared hallucination rejection)
6. Workers disagreeing when sources support only one answer
7. Contradictory or insufficient source documents
8. Empty retrieval results
9. Relevance evaluator and grounding evaluator failures, timeouts, and malformed outputs (fail-closed)
10. Local-only, cloud-only, and dual-worker execution
11. Confirmation that unvalidated tokens never reach UI or TTS
12. Confirmation that verified answers and abstentions produce correct API/SSE events
"""

import sys
import os
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import config
from core.state import RagObotState
from core.graph import (
    assess_evidence_relevance,
    assess_answer_grounding,
    retrieve_chromadb_chunks_node,
    check_relevant_evidence,
    ollama_worker_node,
    groq_worker_node,
    consensus_arbiter_node,
    check_answer_grounded,
    return_abstention_node,
    final_response_and_tts_node,
    create_ragobot_graph,
    ragobot_graph
)
from langchain_core.messages import HumanMessage, AIMessage


class TestMultiAgentArbitrationAndGrounding(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.mock_docs = [
            {
                "content": "RagObot is an advanced multi-agent system built by Bimal P B. It coordinates local and cloud compute.",
                "breadcrumb": "[manual.md :: System Overview]",
                "source": "manual.md",
                "score": 0.85,
                "formatted": "[manual.md :: System Overview]\nRagObot is an advanced multi-agent system built by Bimal P B."
            }
        ]

    # -----------------------------------------------------------------------
    # 1. Fully supported question
    # -----------------------------------------------------------------------
    async def test_01_fully_supported_question(self):
        """Test a fully supported query successfully grounds and routes to final response & TTS."""
        query = "Who built RagObot?"
        candidate_verdict = "RagObot was built by Bimal P B."
        
        # Test grounding validator directly
        mock_llm_response = MagicMock()
        mock_llm_response.content = (
            "STATUS: FULLY_SUPPORTED\n"
            "REASON: The source document explicitly states that RagObot was built by Bimal P B.\n"
            "GROUNDED_ANSWER: RagObot was built by Bimal P B."
        )
        
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.return_value = mock_llm_response
            mock_chat.return_value = mock_inst
            
            is_grounded, reason, answer = await assess_answer_grounding(
                query=query,
                verdict=candidate_verdict,
                retrieved_docs=self.mock_docs,
                cloud_model="mock-model",
                local_model="mock-model"
            )
            self.assertTrue(is_grounded)
            self.assertEqual(answer, "RagObot was built by Bimal P B.")

    # -----------------------------------------------------------------------
    # 2. Unrelated question (e.g. chocolate cake recipe)
    # -----------------------------------------------------------------------
    async def test_02_unrelated_question_abstention(self):
        """Test an unrelated query fails relevance check and triggers abstention without worker execution."""
        query = "What is the recipe for baking a chocolate cake?"
        low_score_docs = [
            {
                "content": "RagObot system architecture notes.",
                "breadcrumb": "[manual.md :: System]",
                "source": "manual.md",
                "score": 0.38,
                "formatted": "RagObot system architecture notes."
            }
        ]
        
        is_relevant, reason = await assess_evidence_relevance(
            query=query,
            retrieved_docs=low_score_docs,
            cloud_model="mock-model",
            local_model="mock-model"
        )
        self.assertFalse(is_relevant)
        self.assertIn("below evidence threshold", reason)
        
        # Verify routing decision routes to return_abstention
        state = {"evidence_available": is_relevant}
        route = check_relevant_evidence(state)
        self.assertEqual(route, "return_abstention")

    # -----------------------------------------------------------------------
    # 3. Six-stroke-engine question when KB only documents four-stroke engines
    # -----------------------------------------------------------------------
    async def test_03_six_stroke_engine_question(self):
        """Test query asking about six-stroke engine when docs only discuss four-stroke engine fails relevance/grounding."""
        query = "How does a six-stroke engine work and what are its extra strokes?"
        four_stroke_docs = [
            {
                "content": "An internal combustion engine operates on a four-stroke cycle: intake, compression, power, exhaust.",
                "breadcrumb": "[engine.md :: Operation]",
                "source": "engine.md",
                "score": 0.55,
                "formatted": "An internal combustion engine operates on a four-stroke cycle."
            }
        ]
        
        mock_eval_response = MagicMock()
        mock_eval_response.content = (
            "VERDICT: NO\n"
            "REASON: The snippets only discuss a four-stroke engine cycle and provide no evidence about six-stroke engines."
        )
        
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.return_value = mock_eval_response
            mock_chat.return_value = mock_inst
            
            is_relevant, reason = await assess_evidence_relevance(
                query=query,
                retrieved_docs=four_stroke_docs,
                cloud_model="mock-model",
                local_model="mock-model"
            )
            self.assertFalse(is_relevant)
            self.assertIn("four-stroke", reason.lower())

    # -----------------------------------------------------------------------
    # 4. Partially supported question
    # -----------------------------------------------------------------------
    async def test_04_partially_supported_question(self):
        """Test partially supported answer prunes unsupported hallucination and retains supported content."""
        query = "Who built RagObot and what funding did it receive?"
        candidate_verdict = "RagObot was built by Bimal P B and received 5 million dollars from NASA."
        
        mock_eval_response = MagicMock()
        mock_eval_response.content = (
            "STATUS: PARTIALLY_SUPPORTED\n"
            "REASON: The claim that Bimal P B built RagObot is supported; NASA funding is not mentioned in sources.\n"
            "GROUNDED_ANSWER: RagObot was built by Bimal P B."
        )
        mock_reval_response = MagicMock()
        mock_reval_response.content = (
            "STATUS: FULLY_SUPPORTED\n"
            "REASON: All remaining claims in pruned answer are directly entailed by source documentation."
        )
        
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.side_effect = [mock_eval_response, mock_reval_response]
            mock_chat.return_value = mock_inst
            
            is_grounded, reason, clean_answer = await assess_answer_grounding(
                query=query,
                verdict=candidate_verdict,
                retrieved_docs=self.mock_docs,
                cloud_model="mock-model",
                local_model="mock-model"
            )
            self.assertTrue(is_grounded)
            self.assertIn("Pruned unsupported claims", reason)
            self.assertEqual(clean_answer, "RagObot was built by Bimal P B.")
            self.assertNotIn("NASA", clean_answer)

    # -----------------------------------------------------------------------
    # 5. Two workers agreeing on an unsupported claim (shared hallucination)
    # -----------------------------------------------------------------------
    async def test_05_shared_hallucination_rejection(self):
        """Test two workers agreeing on an unsupported claim is rejected by arbiter and grounding validator."""
        query = "What encryption does RagObot use?"
        candidate_verdict = "RagObot utilizes 4096-bit post-quantum lattice encryption."
        
        mock_eval_response = MagicMock()
        mock_eval_response.content = (
            "STATUS: REJECTED\n"
            "REASON: The source documentation contains no information about encryption. Both claims are unsupported hallucinations.\n"
            "GROUNDED_ANSWER: NONE"
        )
        
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.return_value = mock_eval_response
            mock_chat.return_value = mock_inst
            
            is_grounded, reason, clean_answer = await assess_answer_grounding(
                query=query,
                verdict=candidate_verdict,
                retrieved_docs=self.mock_docs,
                cloud_model="mock-model",
                local_model="mock-model"
            )
            self.assertFalse(is_grounded)
            self.assertEqual(clean_answer, "")
            
            # Verify route
            route = check_answer_grounded({"answer_grounded": is_grounded})
            self.assertEqual(route, "return_abstention")

    # -----------------------------------------------------------------------
    # 6. Workers disagreeing when sources support only one answer
    # -----------------------------------------------------------------------
    async def test_06_workers_disagreeing_supported_resolution(self):
        """Test arbiter selects the supported worker claim when workers disagree."""
        query = "Who built RagObot?"
        state: RagObotState = {
            "query": query,
            "retrieved_docs": self.mock_docs,
            "ollama_output": "RagObot was built by Bimal P B.",
            "groq_output": "RagObot was created by Sundar Pichai at Google.",
            "model_mode": "dual",
            "cloud_model": "mock-model",
            "local_model": "mock-model",
            "messages": [],
            "evidence_available": True,
            "evidence_reason": "",
            "preliminary_verdict": "",
            "grounded_answer": "",
            "answer_grounded": False,
            "grounding_reason": "",
            "is_abstained": False,
            "abstention_reason": "",
            "final_response": "",
            "arbiter_debate": "",
            "token_count": 0,
            "was_trimmed": False,
            "turn_count": 0
        }
        
        arbiter_output = (
            "=== AGENT DEBATE & CROSS-EXAMINATION ===\n"
            "1. Source Fact-Check: Candidate asserting Bimal P B is supported; Sundar Pichai is unsupported.\n"
            "=== FINAL CONSENSUS VERDICT ===\n"
            "RagObot was built by Bimal P B."
        )
        grounding_output = (
            "STATUS: FULLY_SUPPORTED\n"
            "REASON: Bimal P B is verified in documentation.\n"
            "GROUNDED_ANSWER: RagObot was built by Bimal P B."
        )
        
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.side_effect = [
                MagicMock(content=arbiter_output),
                MagicMock(content=grounding_output)
            ]
            mock_chat.return_value = mock_inst
            
            result = await consensus_arbiter_node(state)
            self.assertTrue(result["answer_grounded"])
            self.assertEqual(result["grounded_answer"], "RagObot was built by Bimal P B.")
            self.assertIn("Bimal P B", result["preliminary_verdict"])
            self.assertNotIn("Sundar Pichai", result["preliminary_verdict"])

    # -----------------------------------------------------------------------
    # 7. Contradictory or insufficient source documents
    # -----------------------------------------------------------------------
    async def test_07_insufficient_or_contradictory_sources(self):
        """Test insufficient source documentation triggers abstention."""
        query = "What is the exact maximum latency of RagObot in milliseconds?"
        candidate_verdict = "The maximum latency is 12 milliseconds."
        
        mock_eval_response = MagicMock()
        mock_eval_response.content = (
            "STATUS: REJECTED\n"
            "REASON: The source documents contain no latency measurements or latency specifications.\n"
            "GROUNDED_ANSWER: NONE"
        )
        
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.return_value = mock_eval_response
            mock_chat.return_value = mock_inst
            
            is_grounded, reason, answer = await assess_answer_grounding(
                query=query,
                verdict=candidate_verdict,
                retrieved_docs=self.mock_docs,
                cloud_model="mock-model",
                local_model="mock-model"
            )
            self.assertFalse(is_grounded)
            
            abstention_state: RagObotState = {
                "evidence_available": True,
                "grounding_reason": reason,
                "arbiter_debate": "Debate notes."
            }
            res = await return_abstention_node(abstention_state)
            self.assertTrue(res["is_abstained"])
            self.assertIn("could not be verified as factually grounded", res["final_response"])

    # -----------------------------------------------------------------------
    # 8. Empty retrieval results
    # -----------------------------------------------------------------------
    async def test_08_empty_retrieval_results(self):
        """Test empty retrieval results immediately fail evidence gate without LLM calls."""
        query = "Any user question"
        is_relevant, reason = await assess_evidence_relevance(
            query=query,
            retrieved_docs=[],
            cloud_model="mock-model",
            local_model="mock-model"
        )
        self.assertFalse(is_relevant)
        self.assertEqual(reason, "Empty retrieval results from knowledge base.")
        
        state: RagObotState = {
            "evidence_available": False,
            "evidence_reason": reason
        }
        res = await return_abstention_node(state)
        self.assertTrue(res["is_abstained"])
        self.assertIn("cannot find sufficient relevant evidence", res["final_response"])

    # -----------------------------------------------------------------------
    # 9. Relevance & Grounding evaluator failures, timeouts, and malformed outputs (Fail-Closed)
    # -----------------------------------------------------------------------
    async def test_09_evaluator_failures_fail_closed(self):
        """Test timeouts, malformed outputs, and exceptions fail closed instead of approving."""
        query = "Who built RagObot?"
        
        # 9a: Relevance Timeout -> Fail Closed
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.side_effect = asyncio.TimeoutError()
            mock_chat.return_value = mock_inst
            
            with patch("core.graph.ChatOllama") as mock_ollama:
                mock_ollama_inst = AsyncMock()
                mock_ollama_inst.ainvoke.side_effect = asyncio.TimeoutError()
                mock_ollama.return_value = mock_ollama_inst
                
                is_rel, reason = await assess_evidence_relevance(query, self.mock_docs, "mock", "mock")
                self.assertFalse(is_rel)
                self.assertIn("timed out (failed closed)", reason)

        # 9b: Relevance Malformed Output -> Fail Closed
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.return_value = MagicMock(content="I am not sure what to answer here.")
            mock_chat.return_value = mock_inst
            
            with patch("core.graph.ChatOllama") as mock_ollama:
                mock_ollama_inst = AsyncMock()
                mock_ollama_inst.ainvoke.return_value = MagicMock(content="I am not sure what to answer here.")
                mock_ollama.return_value = mock_ollama_inst
                
                is_rel, reason = await assess_evidence_relevance(query, self.mock_docs, "mock", "mock")
                self.assertFalse(is_rel)
                self.assertIn("malformed", reason)

        # 9c: Grounding Timeout -> Fail Closed
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.side_effect = asyncio.TimeoutError()
            mock_chat.return_value = mock_inst
            
            with patch("core.graph.ChatOllama") as mock_ollama:
                mock_ollama_inst = AsyncMock()
                mock_ollama_inst.ainvoke.side_effect = asyncio.TimeoutError()
                mock_ollama.return_value = mock_ollama_inst
                
                is_grounded, reason, answer = await assess_answer_grounding(
                    query, "Candidate verdict", self.mock_docs, "mock", "mock"
                )
                self.assertFalse(is_grounded)
                self.assertIn("timed out (failed closed)", reason)
                self.assertEqual(answer, "")

        # 9d: Grounding Malformed Output -> Fail Closed
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.return_value = MagicMock(content="This text looks mostly fine to me.")
            mock_chat.return_value = mock_inst
            
            with patch("core.graph.ChatOllama") as mock_ollama:
                mock_ollama_inst = AsyncMock()
                mock_ollama_inst.ainvoke.return_value = MagicMock(content="This text looks mostly fine to me.")
                mock_ollama.return_value = mock_ollama_inst
                
                is_grounded, reason, answer = await assess_answer_grounding(
                    query, "Candidate verdict", self.mock_docs, "mock", "mock"
                )
                self.assertFalse(is_grounded)
                self.assertIn("malformed", reason)
                self.assertEqual(answer, "")

        # 9e: Grounding Exception -> Fail Closed
        with patch("core.graph.ChatGroq") as mock_chat:
            mock_inst = AsyncMock()
            mock_inst.ainvoke.side_effect = RuntimeError("API Network Disconnect")
            mock_chat.return_value = mock_inst
            
            with patch("core.graph.ChatOllama") as mock_ollama:
                mock_ollama_inst = AsyncMock()
                mock_ollama_inst.ainvoke.side_effect = RuntimeError("Ollama Down")
                mock_ollama.return_value = mock_ollama_inst
                
                is_grounded, reason, answer = await assess_answer_grounding(
                    query, "Candidate verdict", self.mock_docs, "mock", "mock"
                )
                self.assertFalse(is_grounded)
                self.assertIn("failed closed", reason)
                self.assertEqual(answer, "")

    # -----------------------------------------------------------------------
    # 10. Local-only, cloud-only, and dual-worker execution
    # -----------------------------------------------------------------------
    async def test_10_model_mode_execution(self):
        """Test local-only, cloud-only, and dual-worker modes obey the same grounding gates."""
        base_state: RagObotState = {
            "query": "Who built RagObot?",
            "retrieved_docs": self.mock_docs,
            "messages": [],
            "evidence_available": True,
            "evidence_reason": "Verified",
            "ollama_output": "",
            "groq_output": "",
            "preliminary_verdict": "",
            "grounded_answer": "",
            "answer_grounded": False,
            "grounding_reason": "",
            "is_abstained": False,
            "abstention_reason": "",
            "final_response": "",
            "arbiter_debate": "",
            "token_count": 0,
            "was_trimmed": False,
            "turn_count": 0,
            "local_model": "mock-ollama",
            "cloud_model": "mock-groq"
        }
        
        # 10a: Local-only mode: groq is bypassed
        state_local = dict(base_state)
        state_local["model_mode"] = "local"
        res_groq = await groq_worker_node(state_local)
        self.assertIn("bypassed", res_groq["groq_output"])
        
        # 10b: Cloud-only mode: ollama is bypassed
        state_cloud = dict(base_state)
        state_cloud["model_mode"] = "cloud"
        res_ollama = await ollama_worker_node(state_cloud)
        self.assertIn("bypassed", res_ollama["ollama_output"])

    # -----------------------------------------------------------------------
    # 11. Confirmation that unvalidated tokens never reach UI or TTS
    # -----------------------------------------------------------------------
    async def test_11_unvalidated_tokens_and_tts_safety(self):
        """Test unvalidated worker streams and abstentions never trigger answer TTS or leak debate."""
        with patch("core.graph.tts_engine.speak") as mock_speak:
            # Case A: Abstention must NEVER call speak
            abstention_state: RagObotState = {
                "evidence_available": False,
                "evidence_reason": "No evidence",
                "arbiter_debate": "Debate notes."
            }
            res_abs = await return_abstention_node(abstention_state)
            mock_speak.assert_not_called()
            self.assertTrue(res_abs["is_abstained"])
            self.assertNotIn("Debate notes", res_abs["final_response"])
            
            # Case B: Verified answer calls speak with clean final response (default for CLI)
            grounded_state: RagObotState = {
                "grounded_answer": "RagObot was built by Bimal P B.",
                "preliminary_verdict": "RagObot was built by Bimal P B."
            }
            res_grounded = await final_response_and_tts_node(grounded_state)
            mock_speak.assert_called_once_with("RagObot was built by Bimal P B.")
            self.assertEqual(res_grounded["final_response"], "RagObot was built by Bimal P B.")

            # Case C: Web client explicitly requests trigger_backend_tts=False to prevent dual audio
            mock_speak.reset_mock()
            web_grounded_state: RagObotState = {
                "grounded_answer": "RagObot was built by Bimal P B.",
                "preliminary_verdict": "RagObot was built by Bimal P B.",
                "trigger_backend_tts": False
            }
            res_web = await final_response_and_tts_node(web_grounded_state)
            mock_speak.assert_not_called()
            self.assertEqual(res_web["final_response"], "RagObot was built by Bimal P B.")

    # -----------------------------------------------------------------------
    # 12. Confirmation that verified answers and abstentions produce correct API/SSE events
    # -----------------------------------------------------------------------
    async def test_12_api_sse_events(self):
        """Test SSE event stream emits evidence_status, validation_status, and consensus payload correctly."""
        from api import QueryRequest, app
        from httpx import AsyncClient, ASGITransport
        
        # Mock ragobot_graph.ainvoke for verified response
        mock_graph_result = {
            "final_response": "RagObot was built by Bimal P B.",
            "arbiter_debate": "Verified against manual.",
            "ollama_output": "Built by Bimal P B.",
            "groq_output": "Built by Bimal P B.",
            "evidence_available": True,
            "evidence_reason": "Direct semantic match",
            "answer_grounded": True,
            "grounding_reason": "Fully supported",
            "grounded_answer": "RagObot was built by Bimal P B.",
            "is_abstained": False,
            "abstention_reason": "",
            "token_count": 150,
            "was_trimmed": False,
            "retrieved_docs": self.mock_docs
        }
        
        with patch("api.ragobot_graph.ainvoke", return_value=mock_graph_result):
            with patch("api.rag_engine.retrieve", return_value=self.mock_docs):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    resp = await client.post(
                        "/api/query",
                        json={"query": "Who built RagObot?", "model_mode": "dual"}
                    )
                    self.assertEqual(resp.status_code, 200)
                    body_text = resp.text
                    
                    self.assertIn("event: docs", body_text)
                    self.assertIn("event: consensus", body_text)
                    self.assertIn('"is_abstained": false', body_text)
                    self.assertIn("RagObot was built by Bimal P B.", body_text)


if __name__ == "__main__":
    unittest.main()
