"""
Модуль фильтрации контекста поисковой выдачи.
Включает отсечение по порогу сходства, фильтрацию коротких фрагментов и дедупликацию по статьям.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class ContextFilter:
    """
    Класс постобработки и фильтрации чанков после первичного поиска.
    """

    @staticmethod
    def filter_by_threshold(chunks: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
        """
        Отсекает чанки, чей similarity_score ниже заданного порога.
        :param chunks: Список чанков.
        :param threshold: Минимально допустимый similarity_score.
        :return: Отфильтрованный список чанков.
        """
        filtered = [
            chunk for chunk in chunks
            if chunk.get("similarity_score", 0.0) >= threshold
        ]
        logger.debug(f"filter_by_threshold (threshold={threshold}): {len(chunks)} -> {len(filtered)}")
        return filtered

    @staticmethod
    def filter_by_min_length(chunks: list[dict[str, Any]], min_chars: int = 30) -> list[dict[str, Any]]:
        """
        Удаляет обрывки текста и пустые строки короче min_chars символов.
        :param chunks: Список чанков.
        :param min_chars: Минимальное количество непробельных символов в тексте чанка.
        :return: Отфильтрованный список чанков.
        """
        filtered = [
            chunk for chunk in chunks
            if len(chunk.get("text", "").strip()) >= min_chars
        ]
        logger.debug(f"filter_by_min_length (min_chars={min_chars}): {len(chunks)} -> {len(filtered)}")
        return filtered

    @staticmethod
    def deduplicate_by_article(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Устраняет дубликаты одной и той же статьи, оставляя наиболее релевантный чанк.
        (Если чанки не отсортированы по убыванию сходства, выполняется предварительная сортировка).
        :param chunks: Список чанков.
        :return: Дедуплицированный список чанков.
        """
        if not chunks:
            return []

        # Сортируем по убыванию similarity_score, чтобы первым встретился чанк с максимальным баллом
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("similarity_score", 0.0),
            reverse=True,
        )

        seen_articles: set[str] = set()
        deduped: list[dict[str, Any]] = []

        for chunk in sorted_chunks:
            metadata = chunk.get("metadata") or {}
            art_num = metadata.get("article_number")

            if art_num is not None:
                art_key = str(art_num).strip().lower()
                if art_key:
                    if art_key in seen_articles:
                        continue
                    seen_articles.add(art_key)

            deduped.append(chunk)

        logger.debug(f"deduplicate_by_article: {len(chunks)} -> {len(deduped)}")
        return deduped

    @classmethod
    def apply_all(
        cls,
        chunks: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Последовательно применяет все фильтры:
        1. Удаление обрывков и коротких текстов (min_chunk_chars)
        2. Отсечение по порогу сходства (similarity_threshold)
        3. Дедупликация фрагментов одной статьи
        :param chunks: Исходный список найденных чанков.
        :param config: Словарь конфигурации с параметрами фильтрации.
        :return: Результирующий список чанков.
        """
        cfg = config or {}
        min_chars = cfg.get("min_chunk_chars", 30)
        threshold = cfg.get("similarity_threshold", 0.35)

        # 1. Фильтр длины
        result = cls.filter_by_min_length(chunks, min_chars=min_chars)

        # 2. Фильтр порога схожести
        result = cls.filter_by_threshold(result, threshold=threshold)

        # 3. Дедупликация по статьям
        result = cls.deduplicate_by_article(result)

        return result
