"""
Модуль переранжирования документов (Reranking).
Поддерживает:
1. Русскоязычный Cross-Encoder (например, DiTy/cross-encoder-russian-msmarco) через sentence-transformers.
2. Легковесный ONNX FlashRank (ms-marco-MultiBERT-L-12, с алиасом ms-marco-Multi-MiniLM-L-12-v2).
3. Чистый dense-скор fallback (сортировка по вектору схожести ChromaDB).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "retrieval.json"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"

# Маппинг алиасов FlashRank
FLASHRANK_ALIASES = {
    "ms-marco-multi-minilm-l-12-v2": "ms-marco-MultiBERT-L-12",
    "ms-marco-multibert-l-12": "ms-marco-MultiBERT-L-12",
    "multibert": "ms-marco-MultiBERT-L-12",
}


class DocumentReranker:
    """
    Класс для реранкинга найденных документов с поддержкой Cross-Encoder, FlashRank и Dense Fallback.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Инициализация реранкера.
        :param model_name: Имя модели реранкера (по умолчанию из retrieval.json).
        :param cache_dir: Директория для кэширования моделей.
        :param config_path: Путь к файлу конфигурации retrieval.json.
        """
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config = self._load_config(self._config_path)

        self.model_name = model_name or self._config.get("reranker_model", "DiTy/cross-encoder-russian-msmarco")
        self.default_top_n = self._config.get("reranker_top_n", 5)

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = DEFAULT_CACHE_DIR

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._mode = "dense"
        self._cross_encoder = None
        self._flashrank_ranker = None
        self._is_available = False

        self._init_backend()

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить конфиг реранкера {path}: {e}")
        return {}

    def _determine_mode(self, model_name: str) -> str:
        """Определяет режим работы реранкера по имени модели."""
        name_lower = model_name.lower().strip()
        if name_lower in ("dense", "none", "fallback", "disabled", "no_reranker"):
            return "dense"
        if "cross-encoder" in name_lower or "dity" in name_lower or "rubert" in name_lower:
            return "cross_encoder"
        return "flashrank"

    def _init_backend(self) -> None:
        """Инициализация выбранного бэкенда реранкера."""
        mode = self._determine_mode(self.model_name)

        if mode == "dense":
            logger.info("Реранкер настроен в режим 'dense' (ранжирование по similarity_score).")
            self._mode = "dense"
            self._is_available = True
            return

        if mode == "cross_encoder":
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Инициализация Cross-Encoder: model='{self.model_name}'...")
                self._cross_encoder = CrossEncoder(self.model_name, max_length=512)
                self._mode = "cross_encoder"
                self._is_available = True
                logger.info(f"Cross-Encoder '{self.model_name}' успешно загружен.")
                return
            except Exception as e:
                logger.warning(
                    f"Не удалось загрузить Cross-Encoder '{self.model_name}' ({e}). "
                    f"Переключение на fallback dense."
                )
                self._mode = "dense"
                self._is_available = False
                return

        # FlashRank backend
        try:
            from flashrank import Ranker
            actual_flashrank_model = FLASHRANK_ALIASES.get(self.model_name.lower(), self.model_name)
            logger.info(
                f"Инициализация FlashRank: model='{actual_flashrank_model}', cache_dir='{self.cache_dir}'"
            )
            self._flashrank_ranker = Ranker(
                model_name=actual_flashrank_model,
                cache_dir=str(self.cache_dir),
            )
            self._mode = "flashrank"
            self._is_available = True
            logger.info(f"Модель FlashRank '{actual_flashrank_model}' успешно инициализирована.")
        except Exception as e:
            logger.warning(
                f"FlashRank недоступен ({e}). Будет использован fallback по similarity_score."
            )
            self._mode = "dense"
            self._is_available = False

    @property
    def is_available(self) -> bool:
        """Флаг доступности реранкера."""
        return self._is_available

    @property
    def mode(self) -> str:
        """Текущий активный режим реранкера (cross_encoder, flashrank или dense)."""
        return self._mode

    def _fallback_rerank(self, documents: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        """
        Резервный метод ранжирования:
        Сортирует документы по similarity_score и проставляет rerank_score = similarity_score.
        """
        logger.info("Применение fallback-ранжирования по вектору сходства (similarity_score).")
        sorted_docs = sorted(
            documents,
            key=lambda d: d.get("similarity_score", 0.0),
            reverse=True,
        )

        fallback_result = []
        for doc in sorted_docs[:top_n]:
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = float(doc.get("similarity_score", 0.0))
            fallback_result.append(doc_copy)

        return fallback_result

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Переранжирует переданные документы по отношению к поисковому запросу.
        :param query: Поисковый запрос.
        :param documents: Список словарей чанков-кандидатов.
        :param top_n: Количество наилучших документов для возврата (по умолчанию из конфига).
        :return: Топ-N документов с обновленным полем rerank_score, отсортированных по убыванию релевантности.
        """
        n = top_n if top_n is not None else self.default_top_n

        if not documents:
            return []

        if not query or not query.strip():
            logger.warning("Пустой запрос для реранкинга. Возврат исходных документов.")
            return self._fallback_rerank(documents, n)

        # 1. Режим Cross-Encoder
        if self._mode == "cross_encoder" and self._cross_encoder is not None:
            try:
                pairs = [[query.strip(), doc.get("text", "")] for doc in documents]
                raw_scores = self._cross_encoder.predict(pairs)

                reranked_docs: list[dict[str, Any]] = []
                for doc, score in zip(documents, raw_scores):
                    doc_copy = dict(doc)
                    doc_copy["rerank_score"] = round(float(score), 4)
                    reranked_docs.append(doc_copy)

                reranked_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                return reranked_docs[:n]
            except Exception as e:
                logger.warning(f"Ошибка при работе Cross-Encoder ({e}). Переход на fallback.")
                return self._fallback_rerank(documents, n)

        # 2. Режим FlashRank
        if self._mode == "flashrank" and self._flashrank_ranker is not None:
            try:
                from flashrank import RerankRequest

                passages = []
                for doc in documents:
                    passage = dict(doc)
                    passage["id"] = doc.get("id")
                    passage["text"] = doc.get("text", "")
                    passages.append(passage)

                rerank_request = RerankRequest(query=query.strip(), passages=passages)
                results = self._flashrank_ranker.rerank(rerank_request)

                reranked_docs = []
                for item in results:
                    doc_copy = dict(item)
                    doc_copy["rerank_score"] = round(float(item.get("score", 0.0)), 4)
                    reranked_docs.append(doc_copy)

                reranked_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                return reranked_docs[:n]
            except Exception as e:
                logger.warning(f"Ошибка при работе FlashRank ({e}). Переход на fallback.")
                return self._fallback_rerank(documents, n)

        # 3. Чистый dense fallback
        return self._fallback_rerank(documents, n)
