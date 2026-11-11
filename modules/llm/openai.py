import os
from typing import List, Dict
from modules.llm.base import BaseLLMProvider
import openai

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = None, temperature: float = 0.2):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
            
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_answer(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI-powered automobile knowledge assistant.\n"
                    "Answer the user's question using only the retrieved context from the uploaded automobile documents.\n"
                    "The retrieved documents are the primary source of truth.\n"
                    "Do not invent facts, specifications, procedures, or citations.\n"
                    "If the retrieved context does not contain enough information to answer the question, say:\n"
                    "\"I could not find sufficient information in the uploaded automobile documents to answer this question.\"\n"
                    "Give clear and easy-to-understand explanations.\n"
                    "When information is supported by the retrieved documents, provide source references.\n"
                    "If multiple documents are used, identify the relevant documents."
                )
            }
        ]
        
        if conversation_history and isinstance(conversation_history, list):
            hist_items = [m for m in conversation_history if isinstance(m, dict)]
            # Filter out active query if passed at the end of history to prevent duplication
            if hist_items and hist_items[-1].get("role") == "user" and (hist_items[-1].get("content") or "").strip() == query.strip():
                hist_items = hist_items[:-1]

            for msg in hist_items:
                role = "assistant" if msg.get("role") == "assistant" else "user"
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                messages.append({"role": role, "content": content})
                
        prompt = f"Retrieved Context:\n{context}\n\nQuestion:\n{query}"
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer with OpenAI: {str(e)}"
