"""
Модуль семантического кэширования ответов RAG на базе SQLite.
Позволяет избегать повторных обращений к векторному индексу и LLM
для семантически эквивалентных или очень похожих запросов (cosine similarity >= 0.92).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "configs"
    / "guardrails_cache.json"
)


class SemanticCache:
    """
    Семантический кэш вопросно-ответных пар на базе SQLite.
    Хранит векторные представления запросов и возвращает готовый ответ,
    если косинусное сходство нового вопроса превышает заданный порог.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        db_path: str | Path | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        """
        Инициализация кэша и таблицы SQLite.

        :param config_path: Путь к файлу configs/guardrails_cache.json.
        :param db_path: Путь к файлу БД SQLite (переопределяет конфиг).
        :param similarity_threshold: Порог косинусного сходства (переопределяет конфиг, дефолт 0.92).
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config(self.config_path)
        cache_cfg = self.config.get("cache", {})

        self.enabled: bool = cache_cfg.get("enabled", True)
        self.similarity_threshold: float = float(
            similarity_threshold
            if similarity_threshold is not None
            else cache_cfg.get("similarity_threshold", 0.92)
        )

        cfg_db_path = db_path or cache_cfg.get("db_path", "data/cache/semantic_cache.db")
        self.db_path = Path(cfg_db_path)
        if not self.db_path.is_absolute():
            # Относительно корня репозитория
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.db_path = repo_root / self.db_path

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        """Загрузка конфигурации кэша."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации кэша {path}: {e}")
        return {}

    def _get_connection(self) -> sqlite3.Connection:
        """Создание соединения с SQLite."""
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def _init_db(self) -> None:
        """Создание таблицы semantic_cache, если она еще не создана."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    response_text TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()

    def get(
        self,
        query_embedding: list[float] | np.ndarray,
    ) -> dict[str, Any] | None:
        """
        Поиск наиболее близкого ответа в кэше по векторному сходству.

        :param query_embedding: L2-нормализованный эмбеддинг запроса.
        :return: Словарь с кэшированным ответом и метаданными, либо None при cache miss.
        """
        if not self.enabled or query_embedding is None:
            return None

        q_vec = np.array(query_embedding, dtype=np.float32)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, query_text, embedding_blob, response_text, sources_json, hit_count FROM semantic_cache"
            )
            rows = cursor.fetchall()

            if not rows:
                return None

            best_sim = -1.0
            best_row = None

            for row in rows:
                cached_vec = np.frombuffer(row[2], dtype=np.float32)
                sim = float(np.dot(q_vec, cached_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_row = row

            if best_row is not None and best_sim >= self.similarity_threshold:
                row_id = best_row[0]
                matched_query = best_row[1]
                response_text = best_row[3]
                sources = json.loads(best_row[4])
                current_hits = best_row[5] + 1

                # Обновление счетчика попаданий и временной метки
                cursor.execute(
                    "UPDATE semantic_cache SET hit_count = ?, created_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (current_hits, row_id),
                )
                conn.commit()

                logger.info(
                    f"Semantic Cache HIT: similarity={best_sim:.4f} >= {self.similarity_threshold:.2f} "
                    f"(Matched: '{matched_query}', hits={current_hits})"
                )

                return {
                    "response_text": response_text,
                    "sources": sources,
                    "similarity": round(best_sim, 4),
                    "matched_query": matched_query,
                    "hit_count": current_hits,
                    "cache_id": row_id,
                }

        logger.debug(
            f"Semantic Cache MISS: максимальное сходство {best_sim:.4f} < {self.similarity_threshold:.2f}"
        )
        return None

    def set(
        self,
        query: str,
        query_embedding: list[float] | np.ndarray,
        response: str,
        sources: list[str],
    ) -> None:
        """
        Сохранение нового успешного ответа и источников в кэш.

        :param query: Текст пользовательского вопроса.
        :param query_embedding: Эмбеддинг вопроса.
        :param response: Сгенерированный ответ LLM.
        :param sources: Список нормативных источников.
        """
        if not self.enabled or not query or not response:
            return

        blob = np.array(query_embedding, dtype=np.float32).tobytes()
        sources_json = json.dumps(sources, ensure_ascii=False)

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO semantic_cache (query_text, embedding_blob, response_text, sources_json, hit_count)
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (query.strip(), blob, response.strip(), sources_json),
                )
                conn.commit()
                logger.info(f"Сохранена новая запись в Semantic Cache: '{query.strip()[:60]}...'")
        except Exception as e:
            logger.warning(f"Ошибка сохранения в Semantic Cache: {e}")

    def clear(self) -> None:
        """Очистка всех записей кэша."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM semantic_cache")
            conn.commit()
            logger.info("Semantic Cache полностью очищен.")

    def count(self) -> int:
        """Количество записей в кэше."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM semantic_cache")
            return int(cursor.fetchone()[0])
