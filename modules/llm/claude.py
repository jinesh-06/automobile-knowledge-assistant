import os
from typing import List, Dict
from modules.llm.base import BaseLLMProvider
import anthropic

class ClaudeProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = None, temperature: float = 0.2):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
            
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_answer(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        system_prompt = (
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
        
        messages = []
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
                if messages and messages[-1]["role"] == role:
                    # Merge consecutive same-role messages to satisfy Anthropic alternating-role requirement
                    messages[-1]["content"] += f"\n\n{content}"
                else:
                    messages.append({"role": role, "content": content})

            # Anthropic requires the first message to be from 'user'
            while messages and messages[0]["role"] != "user":
                messages.pop(0)

        prompt = f"Retrieved Context:\n{context}\n\nQuestion:\n{query}"
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n\n{prompt}"
        else:
            messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating answer with Claude: {str(e)}"
