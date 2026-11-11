import unittest
from unittest.mock import patch, MagicMock
import os

class TestGeminiConfig(unittest.TestCase):
    """
    Unit tests for Gemini fallback model configuration and environment overrides.
    No live API calls are made.
    """

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_default_fallback_model(self, mock_configure, mock_model_cls):
        """Verify fallback defaults to verified 'gemini-2.0-flash' when env var is absent."""
        env_without_gemini_model = os.environ.copy()
        env_without_gemini_model.pop("GEMINI_MODEL", None)
        env_without_gemini_model["GEMINI_API_KEY"] = "test-key"

        with patch.dict(os.environ, env_without_gemini_model, clear=True):
            from modules.llm.gemini import GeminiProvider
            provider = GeminiProvider(api_key="test-key")
            self.assertEqual(provider.model, "gemini-2.0-flash")
            mock_model_cls.assert_called_once()
            self.assertEqual(mock_model_cls.call_args[1]["model_name"], "gemini-2.0-flash")

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_env_variable_override(self, mock_configure, mock_model_cls):
        """Verify GEMINI_MODEL env var overrides the fallback."""
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-1.5-flash", "GEMINI_API_KEY": "test-key"}):
            from modules.llm.gemini import GeminiProvider
            provider = GeminiProvider(api_key="test-key")
            self.assertEqual(provider.model, "gemini-1.5-flash")
            self.assertEqual(mock_model_cls.call_args[1]["model_name"], "gemini-1.5-flash")

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_explicit_parameter_override(self, mock_configure, mock_model_cls):
        """Verify explicit model parameter takes precedence over env var."""
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-1.5-pro", "GEMINI_API_KEY": "test-key"}):
            from modules.llm.gemini import GeminiProvider
            provider = GeminiProvider(api_key="test-key", model="gemini-2.0-flash-lite")
            self.assertEqual(provider.model, "gemini-2.0-flash-lite")
            self.assertEqual(mock_model_cls.call_args[1]["model_name"], "gemini-2.0-flash-lite")

    def test_gemini_missing_api_key_raises_error(self):
        """Verify missing API key raises ValueError safely without exposing secrets."""
        env_clean = os.environ.copy()
        env_clean.pop("GEMINI_API_KEY", None)

        with patch.dict(os.environ, env_clean, clear=True):
            from modules.llm.gemini import GeminiProvider
            with self.assertRaises(ValueError) as ctx:
                GeminiProvider(api_key=None)
            self.assertIn("GEMINI_API_KEY is not set", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
