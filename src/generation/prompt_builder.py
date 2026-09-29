"""
Модуль форматирования контекста и формирования пользовательских сообщений для LLM.
Отвечает за сборку реранжированных чанков нормативных актов в структурированный контекст.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Класс для подготовки контекста и сообщений в генеративную модель Gemini.
    Форматирует фрагменты статей с разделителями и заголовками для исключения путаницы и галлюцинаций.
    """

    def __init__(self) -> None:
        pass

    def build_context(self, documents: list[dict[str, Any]]) -> str:
        """
        Формирует блок контекста из отобранных документов с четкими разделителями.

        Формат блока для каждого документа:
        [Фрагмент {i+1} | Статья {number}: {title}]
        {text}

        :param documents: Список словарей документов (чанков), отобранных реранкером.
        :return: Текстовый блок контекста.
        """
        if not documents:
            return ""

        context_blocks: list[str] = []
        for i, doc in enumerate(documents):
            meta = doc.get("metadata")
            if not isinstance(meta, dict):
                meta = {}

            # Извлечение номера статьи и заголовка
            number = meta.get("article_number") or doc.get("article_number", "—")
            title = meta.get("title") or doc.get("title", "Без названия")
            text = (doc.get("text") or "").strip()

            block = f"[Фрагмент {i+1} | Статья {number}: {title}]\n{text}\n"
            context_blocks.append(block)

        return "\n".join(context_blocks).strip()

    def build_user_message(self, query: str, documents: list[dict[str, Any]]) -> str:
        """
        Объединяет блок КОНТЕКСТ и ВОПРОС ПОЛЬЗОВАТЕЛЯ в единое сообщение.

        :param query: Вопрос пользователя.
        :param documents: Список отобранных реранкером документов.
        :return: Сформированное сообщение для передачи в LLM.
        """
        clean_query = query.strip()
        context = self.build_context(documents)

        if not context:
            context = "В предоставленном контексте нет релевантных статей или фрагментов."

        user_message = (
            "КОНТЕКСТ:\n"
            f"{context}\n\n"
            "ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n"
            f"{clean_query}"
        )
        return user_message

    @staticmethod
    def extract_sources_from_docs(documents: list[dict[str, Any]]) -> list[str]:
        """
        Извлекает список уникальных источников (статей) из переданных документов.

        :param documents: Список словарей документов.
        :return: Список строк вида: ['Статья 113. Продолжительность ежегодного оплачиваемого отпуска', ...]
        """
        sources: list[str] = []
        seen: set[str] = set()

        for doc in documents:
            meta = doc.get("metadata")
            if not isinstance(meta, dict):
                meta = {}

            art_num = meta.get("article_number") or doc.get("article_number")
            title = meta.get("title") or doc.get("title")

            if art_num:
                if title:
                    src_str = f"Статья {art_num}. {title}"
                else:
                    src_str = f"Статья {art_num}"

                if src_str not in seen:
                    seen.add(src_str)
                    sources.append(src_str)

        return sources
