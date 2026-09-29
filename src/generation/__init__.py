"""
Пакет генерации ответов и защиты от галлюцинаций.
Содержит модули:
- llm_client: обращение к Google GenAI SDK (Gemini)
- prompt_builder: сборка контекста и сообщений
- rag_service: сквозной RAG конвейер
"""

from src.generation.llm_client import GeminiClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.rag_service import RAGService

__all__ = ["GeminiClient", "PromptBuilder", "RAGService"]
