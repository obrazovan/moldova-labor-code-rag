"""
Модуль переранжирования документов (Reranking).
Использует легковесную Cross-Encoder модель FlashRank для высокоточного сопоставления запроса и кандидатов.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "retrieval.json"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"


class DocumentReranker:
    """
    Класс для реранкинга найденных документов с использованием FlashRank.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Инициализация реранкера.
        :param model_name: Имя модели реранкера (по умолчанию из retrieval.json: ms-marco-MiniLM-L-12-v2).
        :param cache_dir: Директория для кэширования ONNX модели (по умолчанию models_cache в корне).
        :param config_path: Путь к файлу конфигурации retrieval.json.
        """
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config = self._load_config(self._config_path)

        self.model_name = model_name or self._config.get("reranker_model", "ms-marco-MiniLM-L-12-v2")
        self.default_top_n = self._config.get("reranker_top_n", 5)

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = DEFAULT_CACHE_DIR

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._ranker = None
        self._is_available = False

        self._init_ranker()

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить конфиг реранкера {path}: {e}")
        return {}

    def _init_ranker(self) -> None:
        """
        Инициализирует Ranker из библиотеки flashrank с перехватом ошибок (fallback режим).
        """
        try:
            from flashrank import Ranker
            logger.info(
                f"Инициализация FlashRank: model='{self.model_name}', cache_dir='{self.cache_dir}'"
            )
            self._ranker = Ranker(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
            )
            self._is_available = True
            logger.info("Модель FlashRank успешно инициализирована.")
        except Exception as e:
            logger.warning(
                f"FlashRank недоступен ({e}). Будет использован fallback по similarity_score."
            )
            self._ranker = None
            self._is_available = False

    @property
    def is_available(self) -> bool:
        """Флаг доступности FlashRank."""
        return self._is_available

    def _fallback_rerank(self, documents: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        """
        Резервный метод ранжирования при недоступности FlashRank:
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

        if not self._is_available or self._ranker is None:
            return self._fallback_rerank(documents, n)

        try:
            from flashrank import RerankRequest

            # Формируем список пассажей для FlashRank
            passages = []
            for doc in documents:
                passage = dict(doc)
                passage["id"] = doc.get("id")
                passage["text"] = doc.get("text", "")
                passages.append(passage)

            rerank_request = RerankRequest(query=query.strip(), passages=passages)
            results = self._ranker.rerank(rerank_request)

            reranked_docs: list[dict[str, Any]] = []
            for item in results:
                doc_copy = dict(item)
                # Нормализуем тип балла (float вместо np.float32)
                doc_copy["rerank_score"] = round(float(item.get("score", 0.0)), 4)
                reranked_docs.append(doc_copy)

            # Сортировка по убыванию rerank_score
            reranked_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return reranked_docs[:n]

        except Exception as e:
            logger.warning(f"Ошибка при работе FlashRank ({e}). Переход на fallback.")
            return self._fallback_rerank(documents, n)
