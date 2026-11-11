from modules.llm.base import BaseLLMProvider

class LLMFactory:
    @staticmethod
    def get_provider(provider_name: str, api_key: str = None, model: str = None, temperature: float = 0.2) -> BaseLLMProvider:
        provider_name = provider_name.lower()
        
        if provider_name == "gemini":
            from modules.llm.gemini import GeminiProvider
            return GeminiProvider(api_key=api_key, model=model, temperature=temperature)
        elif provider_name == "openai":
            from modules.llm.openai import OpenAIProvider
            return OpenAIProvider(api_key=api_key, model=model, temperature=temperature)
        elif provider_name == "claude":
            from modules.llm.claude import ClaudeProvider
            return ClaudeProvider(api_key=api_key, model=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM Provider: {provider_name}")
