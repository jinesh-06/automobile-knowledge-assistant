import unittest
from unittest.mock import MagicMock, patch
from modules.rag_pipeline import RAGPipeline
from modules.retriever import Retriever

class TestCitationAndContext(unittest.TestCase):
    """
    Focused unit tests for citation integrity, same-page chunk preservation,
    metadata resilience, contextual follow-up query rewriting, and comparison guardrails.
    """

    def setUp(self):
        self.mock_retriever = MagicMock()
        # Mock LLM factory to avoid any external API initialization
        with patch("modules.llm.factory.LLMFactory.get_provider") as mock_factory:
            self.mock_llm = MagicMock()
            mock_factory.return_value = self.mock_llm
            self.pipeline = RAGPipeline(
                retriever=self.mock_retriever,
                llm_provider="gemini"
            )

    def test_same_page_multiple_chunks_preserved(self):
        """Verify that multiple distinct chunks from the same page are both preserved in sources and context."""
        retrieved_chunks = [
            {
                "id": "hash_p12_c0",
                "text": "The engine oil capacity is 4.5 liters.",
                "relevance": 0.88,
                "metadata": {
                    "document_name": "camry_manual.pdf",
                    "page_number": 12,
                    "chunk_id": "hash_p12_c0"
                }
            },
            {
                "id": "hash_p12_c1",
                "text": "Recommended viscosity is SAE 0W-20 fully synthetic.",
                "relevance": 0.82,
                "metadata": {
                    "document_name": "camry_manual.pdf",
                    "page_number": 12,
                    "chunk_id": "hash_p12_c1"
                }
            }
        ]

        sources = self.pipeline.extract_sources(retrieved_chunks)
        # Both chunks must be preserved
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["page"], 12)
        self.assertEqual(sources[1]["page"], 12)
        self.assertEqual(sources[0]["chunk_id"], "hash_p12_c0")
        self.assertEqual(sources[1]["chunk_id"], "hash_p12_c1")
        self.assertIn("capacity is 4.5 liters", sources[0]["text"])
        self.assertIn("viscosity is SAE 0W-20", sources[1]["text"])

        # Check context formatting
        context = self.pipeline.format_context(retrieved_chunks)
        self.assertIn("Document: camry_manual.pdf | Page: 12", context)
        self.assertIn("capacity is 4.5 liters", context)
        self.assertIn("viscosity is SAE 0W-20", context)

    def test_duplicate_identical_chunk_ids_deduplicated(self):
        """Identical duplicate chunks should be deduplicated while keeping distinct chunks."""
        retrieved_chunks = [
            {
                "id": "chunk_1",
                "text": "Duplicate chunk text",
                "relevance": 0.9,
                "metadata": {"document_name": "doc.pdf", "page_number": 1, "chunk_id": "chunk_1"}
            },
            {
                "id": "chunk_1",
                "text": "Duplicate chunk text",
                "relevance": 0.9,
                "metadata": {"document_name": "doc.pdf", "page_number": 1, "chunk_id": "chunk_1"}
            },
            {
                "id": "chunk_2",
                "text": "Unique second chunk text",
                "relevance": 0.85,
                "metadata": {"document_name": "doc.pdf", "page_number": 1, "chunk_id": "chunk_2"}
            }
        ]
        sources = self.pipeline.extract_sources(retrieved_chunks)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["chunk_id"], "chunk_1")
        self.assertEqual(sources[1]["chunk_id"], "chunk_2")

    def test_multiple_source_pages_and_multiple_pdfs(self):
        """Verify handling of chunks spanning multiple pages across multiple documents."""
        retrieved_chunks = [
            {
                "id": "c1", "text": "Camry brake spec", "relevance": 0.85,
                "metadata": {"document_name": "Camry.pdf", "page_number": 5, "chunk_id": "c1"}
            },
            {
                "id": "c2", "text": "Camry rotor spec", "relevance": 0.81,
                "metadata": {"document_name": "Camry.pdf", "page_number": 12, "chunk_id": "c2"}
            },
            {
                "id": "c3", "text": "Outback brake spec", "relevance": 0.87,
                "metadata": {"document_name": "Outback.pdf", "page_number": 8, "chunk_id": "c3"}
            }
        ]
        sources = self.pipeline.extract_sources(retrieved_chunks)
        self.assertEqual(len(sources), 3)
        docs = [s["document"] for s in sources]
        pages = [s["page"] for s in sources]
        self.assertEqual(docs, ["Camry.pdf", "Camry.pdf", "Outback.pdf"])
        self.assertEqual(pages, [5, 12, 8])

    def test_missing_or_incomplete_metadata_handled_gracefully(self):
        """Chunks with missing metadata or partial fields must not cause KeyErrors."""
        retrieved_chunks = [
            {"id": "c_none", "text": "Text without metadata", "relevance": 0.7, "metadata": None},
            {"id": "c_empty", "text": "Text with empty metadata", "relevance": 0.65, "metadata": {}},
            {"id": "c_partial", "text": "Text with partial metadata", "relevance": 0.6, "metadata": {"document_name": "doc.pdf"}}
        ]
        sources = self.pipeline.extract_sources(retrieved_chunks)
        self.assertEqual(len(sources), 3)
        self.assertEqual(sources[0]["document"], "Unknown Document")
        self.assertEqual(sources[0]["page"], 1)
        self.assertEqual(sources[1]["document"], "Unknown Document")
        self.assertEqual(sources[2]["document"], "doc.pdf")
        self.assertEqual(sources[2]["page"], 1)

        # Ensure format_context does not crash
        context = self.pipeline.format_context(retrieved_chunks)
        self.assertIn("Unknown Document", context)
        self.assertIn("Text without metadata", context)

    def test_citation_to_context_consistency(self):
        """Every citation source must correspond to a chunk in the formatted context."""
        retrieved_chunks = [
            {"id": "c1", "text": "Passage Alpha", "relevance": 0.9, "metadata": {"document_name": "doc1.pdf", "page_number": 3, "chunk_id": "c1"}},
            {"id": "c2", "text": "Passage Beta", "relevance": 0.8, "metadata": {"document_name": "doc2.pdf", "page_number": 7, "chunk_id": "c2"}}
        ]
        sources = self.pipeline.extract_sources(retrieved_chunks)
        context = self.pipeline.format_context(retrieved_chunks)

        for src in sources:
            self.assertIn(src["document"], context)
            self.assertIn(str(src["page"]), context)
            self.assertIn(src["text"], context)

    def test_empty_retrieval_returns_unsupported_message(self):
        """When retrieval returns zero chunks, return standard unsupported message without calling LLM."""
        self.mock_retriever.retrieve.return_value = []

        answer, sources, chunks = self.pipeline.answer_question("What is the top speed of an F-16?")
        self.assertIn("could not find sufficient information", answer)
        self.assertEqual(sources, [])
        self.assertEqual(chunks, [])
        self.mock_llm.generate_answer.assert_not_called()

    def test_blank_query_handling(self):
        """Blank or whitespace-only queries return prompt to enter question without searching."""
        answer, sources, chunks = self.pipeline.answer_question("   ")
        self.assertIn("Please enter a question", answer)
        self.assertEqual(sources, [])
        self.assertEqual(chunks, [])
        self.mock_retriever.retrieve.assert_not_called()

    def test_contextual_follow_up_retrieval_query_building(self):
        """Verify that normal follow-up questions incorporate prior user query context for retrieval."""
        history = [
            {"role": "user", "content": "What is the engine oil capacity for the 2024 Subaru Outback?"},
            {"role": "assistant", "content": "The 2024 Subaru Outback requires 4.4 liters."}
        ]

        # Case 1: Anaphoric follow-up question
        q1 = "What about the spare tire?"
        retrieval_q1 = self.pipeline.build_retrieval_query(q1, history)
        self.assertIn("What about the spare tire?", retrieval_q1)
        self.assertIn("2024 Subaru Outback", retrieval_q1)

        # Case 2: Short pronoun follow-up
        q2 = "How often should I change it?"
        retrieval_q2 = self.pipeline.build_retrieval_query(q2, history)
        self.assertIn("How often should I change it?", retrieval_q2)
        self.assertIn("2024 Subaru Outback", retrieval_q2)

        # Case 3: Completely new standalone query should NOT be rewritten
        q3 = "What is the recommended tire pressure for the Toyota RAV4 Hybrid?"
        retrieval_q3 = self.pipeline.build_retrieval_query(q3, history)
        self.assertEqual(retrieval_q3, q3)

        # Case 4: No history returns raw query
        retrieval_q4 = self.pipeline.build_retrieval_query(q1, [])
        self.assertEqual(retrieval_q4, q1)

    def test_vehicle_switching_does_not_append_old_vehicle(self):
        """When user switches to a new vehicle, do not append the old vehicle's context."""
        history = [
            {"role": "user", "content": "What is the oil capacity of the Subaru Outback?"},
            {"role": "assistant", "content": "It is 4.4 liters."}
        ]
        new_query = "How about Toyota RAV4?"
        retrieval_query = self.pipeline.build_retrieval_query(new_query, history)
        self.assertEqual(retrieval_query, "How about Toyota RAV4?")
        self.assertNotIn("Subaru", retrieval_query)
        self.assertNotIn("Outback", retrieval_query)

    def test_explicit_new_vehicle_in_followup_does_not_inject_old_vehicle(self):
        """When follow-up explicitly names a new vehicle, do not blindly inject previous vehicle identity."""
        history = [
            {"role": "user", "content": "What is the recommended tire pressure for the Subaru Outback?"},
            {"role": "assistant", "content": "35 PSI front, 33 PSI rear."}
        ]
        new_query = "Does the RAV4 use the same oil type?"
        retrieval_query = self.pipeline.build_retrieval_query(new_query, history)
        self.assertEqual(retrieval_query, "Does the RAV4 use the same oil type?")
        self.assertNotIn("Subaru", retrieval_query)
        self.assertNotIn("Outback", retrieval_query)

    def test_ordinal_references_with_established_vehicles(self):
        """Support ordinal references ('the second one', 'the former', 'the latter') when vehicles are established."""
        history = [
            {"role": "user", "content": "Compare the Subaru Outback and the Toyota RAV4."},
            {"role": "assistant", "content": "Here is the comparison between Outback and RAV4."}
        ]

        # 'the second one' -> Toyota RAV4
        q_second = "What is the oil capacity of the second one?"
        ret_second = self.pipeline.build_retrieval_query(q_second, history)
        self.assertIn("Toyota RAV4", ret_second)
        self.assertIn("What is the oil capacity of the second one?", ret_second)

        # 'the former' -> Subaru Outback
        q_former = "What engine does the former have?"
        ret_former = self.pipeline.build_retrieval_query(q_former, history)
        self.assertIn("Subaru Outback", ret_former)
        self.assertIn("What engine does the former have?", ret_former)

        # 'the latter' -> Toyota RAV4
        q_latter = "What is the fuel tank capacity of the latter?"
        ret_latter = self.pipeline.build_retrieval_query(q_latter, history)
        self.assertIn("Toyota RAV4", ret_latter)

        # 'the other vehicle' -> Toyota RAV4
        q_other = "What about the other vehicle?"
        ret_other = self.pipeline.build_retrieval_query(q_other, history)
        self.assertIn("Toyota RAV4", ret_other)

    def test_ordinal_references_without_established_vehicles_not_invented(self):
        """Do not invent which vehicle an ordinal reference means when conversation does not establish it clearly."""
        history = [
            {"role": "user", "content": "What is the oil capacity of the Subaru Outback?"},
            {"role": "assistant", "content": "It requires 4.4 liters."}
        ]
        # Only one vehicle was mentioned, so 'the second one' is undefined
        q = "What is the oil capacity of the second one?"
        ret = self.pipeline.build_retrieval_query(q, history)
        self.assertEqual(ret, "What is the oil capacity of the second one?")
        self.assertNotIn("Toyota", ret)
        self.assertNotIn("RAV4", ret)

    def test_irrelevant_prior_questions_not_appended(self):
        """Do not indiscriminately append unrelated conversational history (e.g. jokes, greetings) to retrieval queries."""
        # Joke request prior turn
        history_joke = [
            {"role": "user", "content": "Can you tell me a joke about mechanics?"},
            {"role": "assistant", "content": "Why did the mechanic sleep under the car? Because he wanted to wake up oily!"}
        ]
        q1 = "What is the recommended tire pressure?"
        ret1 = self.pipeline.build_retrieval_query(q1, history_joke)
        self.assertEqual(ret1, "What is the recommended tire pressure?")
        self.assertNotIn("joke", ret1)
        self.assertNotIn("mechanics", ret1)

        # Greeting prior turn
        history_greeting = [
            {"role": "user", "content": "Hello, how are you today?"},
            {"role": "assistant", "content": "Hello! I am your AI automobile assistant."}
        ]
        q2 = "What is the engine oil capacity?"
        ret2 = self.pipeline.build_retrieval_query(q2, history_greeting)
        self.assertEqual(ret2, "What is the engine oil capacity?")
        self.assertNotIn("Hello", ret2)

    def test_empty_conversation_history(self):
        """Empty or None conversation history cleanly returns the raw query."""
        q = "What is the tire pressure?"
        self.assertEqual(self.pipeline.build_retrieval_query(q, []), q)
        self.assertEqual(self.pipeline.build_retrieval_query(q, None), q)
        self.assertEqual(self.pipeline.build_retrieval_query(q, "invalid"), q)

    def test_retriever_threshold_filtering_below_equal_above(self):
        """Verify that Retriever correctly filters results below, equal to, and above the threshold."""
        mock_chroma = MagicMock()
        mock_embed = MagicMock()
        mock_embed.get_query_embedding.return_value = [0.1] * 384

        retriever = Retriever(chroma_store=mock_chroma, embedding_manager=mock_embed, default_threshold=0.25)

        # Three chunks with relevance 0.15 (below), 0.25 (equal), 0.35 (above)
        mock_chroma.query.return_value = [
            {"id": "chunk_below", "text": "Low relevance chunk text content", "relevance": 0.15, "metadata": {}},
            {"id": "chunk_equal", "text": "Equal relevance chunk text content", "relevance": 0.25, "metadata": {}},
            {"id": "chunk_above", "text": "High relevance chunk text content", "relevance": 0.35, "metadata": {}}
        ]

        # Case 1: Default threshold 0.25 -> 0.15 discarded, 0.25 retained, 0.35 retained
        results = retriever.retrieve("test query")
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("chunk_equal", result_ids)
        self.assertIn("chunk_above", result_ids)
        self.assertNotIn("chunk_below", result_ids)

        # Case 2: Custom threshold 0.30 -> only 0.35 retained
        results_high = retriever.retrieve("test query", threshold=0.30)
        self.assertEqual(len(results_high), 1)
        self.assertEqual(results_high[0]["id"], "chunk_above")

        # Case 3: Custom threshold 0.40 -> all discarded, empty list returned
        results_empty = retriever.retrieve("test query", threshold=0.40)
        self.assertEqual(results_empty, [])

    def test_pipeline_configurable_threshold_and_override(self):
        """Verify RAGPipeline passes configured threshold and supports per-call override."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with patch("modules.llm.factory.LLMFactory.get_provider") as mock_factory:
            mock_factory.return_value = MagicMock()
            # Pipeline with similarity_threshold=0.35
            pipe = RAGPipeline(retriever=mock_retriever, similarity_threshold=0.35)

            # Standard call uses pipeline similarity_threshold
            pipe.answer_question("query 1")
            called_thresh = mock_retriever.retrieve.call_args[1]["threshold"]
            self.assertEqual(called_thresh, 0.35)

            # Call with explicit override uses the override
            pipe.answer_question("query 2", threshold=0.50)
            called_thresh_override = mock_retriever.retrieve.call_args[1]["threshold"]
            self.assertEqual(called_thresh_override, 0.50)

    def test_compare_vehicles_guardrail_prompt(self):
        """Verify comparison query includes strict instruction against assuming missing specs."""
        self.mock_retriever.retrieve.return_value = [
            {"id": "c1", "text": "Vehicle A engine: 2.5L", "relevance": 0.8, "metadata": {"document_name": "A.pdf", "page_number": 1, "chunk_id": "c1"}}
        ]
        self.mock_llm.generate_answer.return_value = "Comparison Table"

        answer, sources = self.pipeline.compare_vehicles("Compare engine specs", ["A.pdf", "B.pdf"])
        self.assertEqual(answer, "Comparison Table")
        self.assertEqual(len(sources), 1)

        # Inspect prompt passed to LLM
        called_query = self.mock_llm.generate_answer.call_args[1]["query"]
        self.assertIn("Not specified in provided manual", called_query)
        self.assertIn("Do not invent, assume, or extrapolate", called_query)

if __name__ == "__main__":
    unittest.main()
