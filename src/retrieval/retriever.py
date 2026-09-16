"""
Модуль векторного поиска по базе документов ChromaDB.
Подключается к коллекции векторов и выполняет семантический поиск кандидатов.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb

from src.embeddings.embedder import EmbeddingModel, DEFAULT_CONFIG_PATH as DEFAULT_EMBEDDINGS_CFG_PATH


logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "retrieval.json"


class VectorRetriever:
    """
    Класс для выполнения семантического векторного поиска в ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "chunks_by_article_rubert-tiny2",
        persist_directory: str | Path | None = None,
        model_name_or_alias: str = "rubert-tiny2",
        retrieval_config_path: str | Path | None = None,
        embeddings_config_path: str | Path | None = None,
    ) -> None:
        """
        Инициализация векторного поисковика.
        :param collection_name: Имя коллекции в ChromaDB (по умолчанию 'chunks_by_article_rubert-tiny2').
        :param persist_directory: Путь к директории базы данных ChromaDB (если None, берется из embeddings.json).
        :param model_name_or_alias: Алиас или имя модели эмбеддингов для векторизации запросов.
        :param retrieval_config_path: Путь к конфигурации retrieval.json.
        :param embeddings_config_path: Путь к конфигурации embeddings.json.
        """
        self._retrieval_cfg_path = Path(retrieval_config_path) if retrieval_config_path else DEFAULT_RETRIEVAL_CONFIG_PATH
        self._embeddings_cfg_path = Path(embeddings_config_path) if embeddings_config_path else DEFAULT_EMBEDDINGS_CFG_PATH

        self._retrieval_config = self._load_json(self._retrieval_cfg_path)
        self._embeddings_config = self._load_json(self._embeddings_cfg_path)

        # Определение директории хранения ChromaDB
        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            dir_name = self._embeddings_config.get("persist_directory", "chroma_db")
            self.persist_directory = Path(dir_name)

        if not self.persist_directory.is_absolute():
            # Если относительный путь, резолвим от корня проекта
            project_root = Path(__file__).resolve().parent.parent.parent
            self.persist_directory = project_root / self.persist_directory

        self.collection_name = collection_name
        self.default_top_k = self._retrieval_config.get("retriever_top_k", 20)

        logger.info(
            f"Инициализация VectorRetriever: collection='{self.collection_name}', "
            f"chroma_dir='{self.persist_directory}', model='{model_name_or_alias}'"
        )

        # Подключение к локальной ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_collection(name=self.collection_name)

        # Инициализация модели эмбеддингов для векторизации поисковых запросов
        self.embedder = EmbeddingModel(
            model_name_or_alias=model_name_or_alias,
            config_path=self._embeddings_cfg_path,
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось прочитать {path}: {e}")
        return {}

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Выполняет семантический поиск по запросу и возвращает список найденных чанков.
        :param query: Поисковый запрос на естественном языке.
        :param top_k: Количество возвращаемых кандидатов (по умолчанию из configs/retrieval.json).
        :return: Список словарей чанков с полями: id, text, metadata, distance, similarity_score.
        """
        k = top_k if top_k is not None else self.default_top_k
        if not query or not query.strip():
            logger.warning("Передан пустой поисковый запрос.")
            return []

        # Векторизация запроса
        query_embedding = self.embedder.encode_query(query.strip())

        # Запрос к ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(ids)):
                dist = float(distances[i]) if distances else 0.0
                # Для косинусного расстояния: similarity = 1.0 - distance
                similarity_score = round(1.0 - dist, 4)

                chunks.append({
                    "id": ids[i],
                    "text": docs[i] if docs else "",
                    "metadata": metas[i] if metas and metas[i] is not None else {},
                    "distance": dist,
                    "similarity_score": similarity_score,
                })

        # Гарантируем сортировку по убыванию сходства
        chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return chunks
