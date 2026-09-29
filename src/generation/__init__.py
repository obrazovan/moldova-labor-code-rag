"""
Пакет генерации ответов и защиты от галлюцинаций.
Содержит модули:
- llm_client: обращение к Google GenAI SDK (Gemini)
- prompt_builder: сборка контекста и сообщений
- rag_service: сквозной RAG конвейер
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "GeminiClient":
        from src.generation.llm_client import GeminiClient
        return GeminiClient
    if name == "PromptBuilder":
        from src.generation.prompt_builder import PromptBuilder
        return PromptBuilder
    if name == "RAGService":
        from src.generation.rag_service import RAGService
        return RAGService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GeminiClient", "PromptBuilder", "RAGService"]
