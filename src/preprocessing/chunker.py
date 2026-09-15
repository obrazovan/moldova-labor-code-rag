"""Модуль чанкинга законодательных текстов с поддержкой 4 стратегий разбиения.

Стратегии:
1. by_article: статья целиком является одним чанком.
2. fixed_size: фиксированный размер (параметры: chunk_size=500).
3. fixed_overlap: фиксированный размер с нахлестом (chunk_size=500, overlap=100).
4. by_paragraph: разбиение статьи по абзацам/пунктам.

Каждый чанк содержит уникальный chunk_id и метаданные:
source, url, title, section, document_id, updated_at, chunk_strategy.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("preprocessing.chunker")


class TextChunker:
    """Класс для разбиения статей законодательных актов на чанки."""

    STRATEGIES = ("by_article", "fixed_size", "fixed_overlap", "by_paragraph")

    def __init__(self, default_chunk_size: int = 500, default_overlap: int = 100) -> None:
        """Инициализация чанкера с базовыми параметрами размера и перекрытия.

        Args:
            default_chunk_size: Размер чанка по умолчанию в символах (500).
            default_overlap: Размер перекрытия по умолчанию в символах (100).
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    @staticmethod
    def _safe_art_num(num: Any) -> str:
        """Безопасный строковый идентификатор номера статьи для chunk_id."""
        s = str(num).strip()
        s = s.replace("^", "_").replace(" ", "")
        return s or "0"

    def _build_chunk(
        self,
        text: str,
        article: Dict[str, Any],
        strategy: str,
        chunk_index: int = 0,
        chunk_id_suffix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Формирование словаря чанка с уникальным ID и метаданными."""
        doc_id = str(article.get("document_id") or article.get("id") or "codul_muncii")
        source = str(article.get("source") or doc_id)
        url = str(article.get("url") or "")
        title = str(article.get("title") or article.get("document_title") or "")
        section = str(article.get("section") or "")
        chapter = str(article.get("chapter") or "")
        updated_at = str(article.get("downloaded_at") or article.get("updated_at") or "")
        art_num = str(article.get("number") or article.get("article_number") or "")
        safe_num = self._safe_art_num(art_num)

        suffix = chunk_id_suffix if chunk_id_suffix is not None else str(chunk_index)
        if strategy == "by_article":
            chunk_id = f"{doc_id}_{strategy}_art_{safe_num}"
        else:
            chunk_id = f"{doc_id}_{strategy}_art_{safe_num}_{suffix}"

        metadata = {
            "source": source,
            "url": url,
            "title": title,
            "section": section,
            "chapter": chapter,
            "document_id": doc_id,
            "updated_at": updated_at,
            "chunk_strategy": strategy,
            "article_number": art_num,
            "chunk_index": chunk_index,
        }

        return {
            "chunk_id": chunk_id,
            "text": text,
            "source": source,
            "url": url,
            "title": title,
            "section": section,
            "document_id": doc_id,
            "updated_at": updated_at,
            "chunk_strategy": strategy,
            "metadata": metadata,
        }

    def chunk_by_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Стратегия by_article: статья целиком является одним чанком."""
        text = str(article.get("text") or "").strip()
        if not text:
            # Если text пустой, пробуем собрать из заголовка и тела
            art_num = article.get("number", "")
            art_title = article.get("title", "")
            body = article.get("body", "")
            text = f"Статья {art_num}. {art_title}\n\n{body}".strip()

        chunk = self._build_chunk(
            text=text,
            article=article,
            strategy="by_article",
            chunk_index=0,
        )
        return [chunk]

    def chunk_fixed_size(self, article: Dict[str, Any], chunk_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Стратегия fixed_size: разбиение статьи на фрагменты фиксированного размера."""
        size = chunk_size or self.default_chunk_size
        text = str(article.get("text") or "").strip()
        if not text:
            return []

        chunks: List[Dict[str, Any]] = []
        n = len(text)
        idx = 0
        for pos in range(0, n, size):
            chunk_text = text[pos : pos + size].strip()
            if chunk_text:
                chunk = self._build_chunk(
                    text=chunk_text,
                    article=article,
                    strategy="fixed_size",
                    chunk_index=idx,
                )
                chunks.append(chunk)
                idx += 1

        return chunks

    def chunk_fixed_overlap(
        self,
        article: Dict[str, Any],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Стратегия fixed_overlap: разбиение статьи с фиксированным размером и нахлестом."""
        size = chunk_size or self.default_chunk_size
        ov = overlap if overlap is not None else self.default_overlap

        if ov >= size:
            raise ValueError(f"Размер нахлеста overlap ({ov}) должен быть меньше chunk_size ({size})")

        step = size - ov
        text = str(article.get("text") or "").strip()
        if not text:
            return []

        if len(text) <= size:
            chunk = self._build_chunk(
                text=text,
                article=article,
                strategy="fixed_overlap",
                chunk_index=0,
            )
            return [chunk]

        chunks: List[Dict[str, Any]] = []
        n = len(text)
        idx = 0
        for pos in range(0, n, step):
            chunk_text = text[pos : pos + size].strip()
            if chunk_text:
                chunk = self._build_chunk(
                    text=chunk_text,
                    article=article,
                    strategy="fixed_overlap",
                    chunk_index=idx,
                )
                chunks.append(chunk)
                idx += 1
            if pos + size >= n:
                break

        return chunks

    def chunk_by_paragraph(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Стратегия by_paragraph: разбиение статьи по отдельным абзацам и пунктам."""
        paragraphs = article.get("paragraphs")
        chunks: List[Dict[str, Any]] = []

        if paragraphs and isinstance(paragraphs, list) and len(paragraphs) > 0:
            for idx, p in enumerate(paragraphs):
                p_text = str(p).strip()
                if p_text:
                    chunk = self._build_chunk(
                        text=p_text,
                        article=article,
                        strategy="by_paragraph",
                        chunk_index=idx,
                    )
                    chunks.append(chunk)
        else:
            # Если абзацы не были разделены, используем полный текст статьи
            text = str(article.get("text") or "").strip()
            if text:
                chunk = self._build_chunk(
                    text=text,
                    article=article,
                    strategy="by_paragraph",
                    chunk_index=0,
                )
                chunks.append(chunk)

        return chunks

    def chunk_article(self, article: Dict[str, Any], strategy: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Разбиение одной статьи в соответствии с выбранной стратегией."""
        if strategy == "by_article":
            return self.chunk_by_article(article)
        elif strategy == "fixed_size":
            chunk_size = kwargs.get("chunk_size", self.default_chunk_size)
            return self.chunk_fixed_size(article, chunk_size=chunk_size)
        elif strategy == "fixed_overlap":
            chunk_size = kwargs.get("chunk_size", self.default_chunk_size)
            overlap = kwargs.get("overlap", self.default_overlap)
            return self.chunk_fixed_overlap(article, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "by_paragraph":
            return self.chunk_by_paragraph(article)
        else:
            raise ValueError(f"Неизвестная стратегия разбиения '{strategy}'. Доступные: {self.STRATEGIES}")

    def chunk_articles(
        self,
        articles: List[Dict[str, Any]],
        strategy: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Разбиение списка статей в соответствии с выбранной стратегией."""
        all_chunks: List[Dict[str, Any]] = []
        for article in articles:
            art_chunks = self.chunk_article(article, strategy=strategy, **kwargs)
            all_chunks.extend(art_chunks)
        return all_chunks


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_article = {
        "number": "1",
        "title": "Основные понятия",
        "text": "Статья 1. Основные понятия\n\n(1) В настоящем кодексе используются понятия...\n\n(2) Работник - физическое лицо...",
        "paragraphs": [
            "(1) В настоящем кодексе используются понятия...",
            "(2) Работник - физическое лицо...",
        ],
        "section": "РАЗДЕЛ I. ОБЩИЕ ПОЛОЖЕНИЯ",
        "chapter": "Глава I. ВВОДНЫЕ ПОЛОЖЕНИЯ",
        "document_id": "codul_muncii",
        "source": "codul_muncii",
        "url": "https://www.legis.md/cautare/getResults?doc_id=155882&lang=ru",
        "downloaded_at": "2026-09-14T20:22:22.422099+00:00",
    }

    chunker = TextChunker()
    for strat in TextChunker.STRATEGIES:
        res = chunker.chunk_article(test_article, strategy=strat)
        print(f"Стратегия {strat:15}: получено {len(res)} чанков. Первый chunk_id: {res[0]['chunk_id']}")
