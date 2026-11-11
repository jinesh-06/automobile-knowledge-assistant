import os
from typing import List, Dict
from modules.llm.base import BaseLLMProvider

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = None, temperature: float = 0.2):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
            
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        
        # Generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.temperature
        )
        self.client = genai.GenerativeModel(model_name=self.model, generation_config=self.generation_config)

    def generate_answer(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        history_section = ""
        if conversation_history and isinstance(conversation_history, list):
            hist_items = [m for m in conversation_history if isinstance(m, dict)]
            # Filter out active query if passed at the end of history to prevent duplication
            if hist_items and hist_items[-1].get("role") == "user" and (hist_items[-1].get("content") or "").strip() == query.strip():
                hist_items = hist_items[:-1]

            formatted_history = []
            for msg in hist_items:
                role = msg.get("role", "")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                role_label = "Assistant" if role == "assistant" else "User"
                formatted_history.append(f"{role_label}: {content}")
            if formatted_history:
                history_section = "Prior Conversation History:\n" + "\n".join(formatted_history) + "\n\n"

        prompt = f"""You are an AI-powered automobile knowledge assistant.
Answer the user's question using only the retrieved context from the uploaded automobile documents.
The retrieved documents are the primary source of truth.
Do not invent facts, specifications, procedures, or citations.
If the retrieved context does not contain enough information to answer the question, say:
"I could not find sufficient information in the uploaded automobile documents to answer this question."
Give clear and easy-to-understand explanations.
When information is supported by the retrieved documents, provide source references.
If multiple documents are used, identify the relevant documents.

{history_section}Retrieved Context:
{context}

Question:
{query}
"""
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer with Gemini: {str(e)}"
