import unittest
from unittest.mock import MagicMock, patch
import os

class TestConversationHistory(unittest.TestCase):
    """
    Focused unit tests for conversation history handling across providers.
    Uses mocks to prevent any live API calls.
    """

    def setUp(self):
        # Set dummy keys for initialization
        self.patcher_env = patch.dict(os.environ, {
            "GEMINI_API_KEY": "dummy-gemini-key",
            "OPENAI_API_KEY": "dummy-openai-key",
            "ANTHROPIC_API_KEY": "dummy-claude-key"
        })
        self.patcher_env.start()

    def tearDown(self):
        self.patcher_env.stop()

    # --- Gemini Provider Tests ---
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_first_turn(self, mock_configure, mock_model_cls):
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MagicMock(text="Mock Gemini Response")
        mock_model_cls.return_value = mock_instance

        from modules.llm.gemini import GeminiProvider
        provider = GeminiProvider(api_key="dummy")

        answer = provider.generate_answer(
            query="What is the tire pressure?",
            context="Tire pressure: 32 PSI.",
            conversation_history=[]
        )
        self.assertEqual(answer, "Mock Gemini Response")
        prompt_arg = mock_instance.generate_content.call_args[0][0]
        self.assertIn("Question:\nWhat is the tire pressure?", prompt_arg)
        self.assertIn("Retrieved Context:\nTire pressure: 32 PSI.", prompt_arg)
        self.assertNotIn("Prior Conversation History:", prompt_arg)

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_multi_turn_propagation(self, mock_configure, mock_model_cls):
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MagicMock(text="Follow-up answer")
        mock_model_cls.return_value = mock_instance

        from modules.llm.gemini import GeminiProvider
        provider = GeminiProvider(api_key="dummy")

        history = [
            {"role": "user", "content": "What engine oil is used?"},
            {"role": "assistant", "content": "SAE 5W-30 synthetic oil."}
        ]
        provider.generate_answer(
            query="How many quarts?",
            context="Oil capacity: 4.5 quarts.",
            conversation_history=history
        )
        prompt_arg = mock_instance.generate_content.call_args[0][0]
        self.assertIn("Prior Conversation History:", prompt_arg)
        self.assertIn("User: What engine oil is used?", prompt_arg)
        self.assertIn("Assistant: SAE 5W-30 synthetic oil.", prompt_arg)
        self.assertIn("Question:\nHow many quarts?", prompt_arg)

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_avoid_duplicate_current_question(self, mock_configure, mock_model_cls):
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MagicMock(text="Answer")
        mock_model_cls.return_value = mock_instance

        from modules.llm.gemini import GeminiProvider
        provider = GeminiProvider(api_key="dummy")

        # History inadvertently contains the active query at the end
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "Current question"}
        ]
        provider.generate_answer(
            query="Current question",
            context="Some context",
            conversation_history=history
        )
        prompt_arg = mock_instance.generate_content.call_args[0][0]
        # "Current question" should only appear in Question: section, NOT in Prior Conversation History
        history_section = prompt_arg.split("Retrieved Context:")[0]
        self.assertNotIn("User: Current question", history_section)
        self.assertIn("User: Previous question", history_section)

    # --- Claude Provider Tests ---
    @patch("anthropic.Anthropic")
    def test_claude_first_turn(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(content=[MagicMock(text="Claude answer")])
        mock_anthropic_cls.return_value = mock_client

        from modules.llm.claude import ClaudeProvider
        provider = ClaudeProvider(api_key="dummy")

        answer = provider.generate_answer(
            query="What is brake fluid type?",
            context="Use DOT 4.",
            conversation_history=[]
        )
        self.assertEqual(answer, "Claude answer")
        messages_sent = mock_client.messages.create.call_args[1]["messages"]
        self.assertEqual(len(messages_sent), 1)
        self.assertEqual(messages_sent[0]["role"], "user")
        self.assertIn("What is brake fluid type?", messages_sent[0]["content"])

    @patch("anthropic.Anthropic")
    def test_claude_role_sequencing_and_no_consecutive_user_turns(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(content=[MagicMock(text="Claude answer")])
        mock_anthropic_cls.return_value = mock_client

        from modules.llm.claude import ClaudeProvider
        provider = ClaudeProvider(api_key="dummy")

        # Multi-turn history with prior turns
        history = [
            {"role": "user", "content": "Turn 1 question"},
            {"role": "assistant", "content": "Turn 1 answer"},
            # Inadvertent duplicate user turn:
            {"role": "user", "content": "Turn 2 question"}
        ]
        provider.generate_answer(
            query="Turn 2 question",
            context="Context for turn 2",
            conversation_history=history
        )
        messages_sent = mock_client.messages.create.call_args[1]["messages"]
        
        # Verify strict alternating roles
        self.assertGreaterEqual(len(messages_sent), 2)
        for i in range(len(messages_sent) - 1):
            self.assertNotEqual(
                messages_sent[i]["role"],
                messages_sent[i+1]["role"],
                f"Consecutive identical role detected at index {i} and {i+1}: {messages_sent[i]['role']}"
            )
        self.assertEqual(messages_sent[0]["role"], "user")
        self.assertEqual(messages_sent[-1]["role"], "user")

    # --- OpenAI Provider Tests ---
    @patch("openai.OpenAI")
    def test_openai_no_duplicated_final_question(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI answer"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_openai_cls.return_value = mock_client

        from modules.llm.openai import OpenAIProvider
        provider = OpenAIProvider(api_key="dummy")

        history = [
            {"role": "user", "content": "Prior Q"},
            {"role": "assistant", "content": "Prior A"},
            {"role": "user", "content": "Active Q"}  # Duplicate query in history
        ]
        provider.generate_answer(
            query="Active Q",
            context="Some context",
            conversation_history=history
        )
        messages_sent = mock_client.chat.completions.create.call_args[1]["messages"]
        
        # Count occurrences of "Active Q" in user messages
        user_messages = [m for m in messages_sent if m["role"] == "user"]
        # Only the final prompt message should contain "Active Q"
        self.assertEqual(len(user_messages), 2)  # "Prior Q" and final prompt with "Active Q"
        self.assertIn("Prior Q", user_messages[0]["content"])
        self.assertIn("Active Q", user_messages[1]["content"])

    # --- Malformed History and Role Integrity Tests ---
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_malformed_history_handled_gracefully(self, mock_configure, mock_model_cls):
        """Verify Gemini handles non-dict, None, or string history entries without crashing."""
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MagicMock(text="Response")
        mock_model_cls.return_value = mock_instance

        from modules.llm.gemini import GeminiProvider
        provider = GeminiProvider(api_key="dummy")

        # History with mixed malformed types
        malformed_history = [
            None,
            "corrupted string",
            123,
            {"role": "assistant", "content": "Valid assistant reply"},
            {"role": "user", "content": None},  # None content
            {"invalid_key": "no role"},          # missing role
            {"role": "user", "content": "   "}   # empty content
        ]
        answer = provider.generate_answer("Query", "Context", conversation_history=malformed_history)
        self.assertEqual(answer, "Response")
        prompt_arg = mock_instance.generate_content.call_args[0][0]
        # Ensure only the valid assistant reply is formatted, and labeled Assistant
        self.assertIn("Assistant: Valid assistant reply", prompt_arg)
        self.assertNotIn("User: None", prompt_arg)

    @patch("anthropic.Anthropic")
    def test_claude_malformed_history_handled_gracefully(self, mock_anthropic_cls):
        """Verify Claude handles malformed history entries and preserves assistant role."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(content=[MagicMock(text="Claude answer")])
        mock_anthropic_cls.return_value = mock_client

        from modules.llm.claude import ClaudeProvider
        provider = ClaudeProvider(api_key="dummy")

        malformed_history = [
            None,
            "bad string",
            {"role": "user", "content": "User first"},
            {"role": "assistant", "content": "Assistant second"},
            {"role": "assistant", "content": None}  # None content
        ]
        provider.generate_answer("Followup", "Context", conversation_history=malformed_history)
        messages_sent = mock_client.messages.create.call_args[1]["messages"]
        self.assertEqual(len(messages_sent), 3)
        self.assertEqual(messages_sent[0]["role"], "user")
        self.assertEqual(messages_sent[1]["role"], "assistant")
        self.assertEqual(messages_sent[1]["content"], "Assistant second")
        self.assertEqual(messages_sent[2]["role"], "user")

    @patch("openai.OpenAI")
    def test_openai_malformed_history_handled_gracefully(self, mock_openai_cls):
        """Verify OpenAI handles malformed history entries without error."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI answer"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_openai_cls.return_value = mock_client

        from modules.llm.openai import OpenAIProvider
        provider = OpenAIProvider(api_key="dummy")

        malformed_history = [
            None,
            "corrupt string",
            {"role": "user", "content": "Valid prior Q"},
            {"role": "assistant", "content": "Valid prior A"}
        ]
        provider.generate_answer("New Q", "Context", conversation_history=malformed_history)
        messages_sent = mock_client.chat.completions.create.call_args[1]["messages"]
        roles = [m["role"] for m in messages_sent]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    # --- App Message Lifecycle Simulation ---
    def test_app_message_lifecycle_history_slicing(self):
        """Simulate app.py message lifecycle to verify slicing logic."""
        session_messages = []
        
        # Turn 1
        query1 = "Question 1"
        hist1 = [{"role": m["role"], "content": m["content"]} for m in session_messages[-4:]]
        self.assertEqual(hist1, [])  # Empty on turn 1
        
        session_messages.append({"role": "user", "content": query1})
        session_messages.append({"role": "assistant", "content": "Answer 1"})

        # Turn 2
        query2 = "Question 2"
        hist2 = [{"role": m["role"], "content": m["content"]} for m in session_messages[-4:]]
        self.assertEqual(len(hist2), 2)
        self.assertEqual(hist2[0]["content"], "Question 1")
        self.assertEqual(hist2[1]["content"], "Answer 1")
        # Ensure query2 is NOT in hist2
        self.assertNotIn("Question 2", [m["content"] for m in hist2])

if __name__ == "__main__":
    unittest.main()
