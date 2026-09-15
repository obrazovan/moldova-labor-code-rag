"""
Модуль векторной индексации чанков в ChromaDB.
Обеспечивает батчевую векторизацию и сохранение векторов с метаданными.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import chromadb

from src.embeddings.embedder import EmbeddingModel, DEFAULT_CONFIG_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class VectorIndexer:
    """
    Класс для управления локальным векторным индексом ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Инициализация векторного индексатора.
        :param persist_directory: Директория локальной БД ChromaDB.
                                  Если None, считывается из configs/embeddings.json.
        :param config_path: Путь к конфигурационному файлу (по умолчанию configs/embeddings.json).
        """
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config = self._load_config(self._config_path)

        # Определение директории сохранения
        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            dir_from_cfg = self._config.get("persist_directory", "chroma_db")
            self.persist_directory = Path(dir_from_cfg)

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Подключение к ChromaDB (директория: {self.persist_directory})")
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось прочитать конфигурацию {path}: {e}")
                return {}
        return {}

    def get_or_create_collection(
        self,
        collection_name: str,
        space: str = "cosine",
    ) -> chromadb.Collection:
        """
        Получение или создание коллекции с заданной метрикой расстояния.
        :param collection_name: Имя коллекции в ChromaDB.
        :param space: Метрика расстояния (по умолчанию cosine).
        :return: Объект коллекции chromadb.Collection.
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": space},
        )

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """
        Нормализация метаданных для ChromaDB: исключение None и сложных вложенных типов.
        """
        clean_meta: dict[str, str | int | float | bool] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        return clean_meta

    def index_chunks(
        self,
        chunks_file_path: str | Path,
        model_alias: str = "rubert-tiny2",
        collection_name: str | None = None,
        batch_size: int | None = None,
        reset_collection: bool = False,
    ) -> dict[str, Any]:
        """
        Загрузка чанков из JSON, батчевая векторизация и сохранение в ChromaDB.
        :param chunks_file_path: Путь к JSON файлу со списком чанков.
        :param model_alias: Алиас модели эмбеддингов (rubert-tiny2, multilingual-e5-small).
        :param collection_name: Имя целевой коллекции (по умолчанию {file_stem}_{model_alias}).
        :param batch_size: Размер батча (если None, берется из конфига или 32).
        :param reset_collection: Если True, удаляет существующую коллекцию перед индексацией.
        :return: Словарь со статистикой индексации.
        """
        chunks_path = Path(chunks_file_path)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Файл чанков не найден: {chunks_path}")

        logger.info(f"Чтение чанков из: {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks: list[dict[str, Any]] = json.load(f)

        total_chunks = len(chunks)
        if total_chunks == 0:
            logger.warning("Файл чанков пуст. Индексация отменена.")
            return {"total_chunks": 0}

        # Определение имени коллекции
        if collection_name is None:
            collection_name = f"{chunks_path.stem}_{model_alias}".replace(".", "_")

        if reset_collection:
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Существующая коллекция '{collection_name}' удалена (reset=True).")
            except Exception:
                pass

        collection = self.get_or_create_collection(collection_name=collection_name)

        # Инициализация модели эмбеддингов
        logger.info(f"Инициализация модели эмбеддингов: {model_alias}")
        embedder = EmbeddingModel(
            model_name_or_alias=model_alias,
            config_path=self._config_path,
        )

        bs = batch_size or self._config.get("batch_size", 32)
        logger.info(
            f"Старт векторизации: {total_chunks} чанков | batch_size={bs} | "
            f"модель={embedder.model_name} (dim={embedder.dimension}) | "
            f"коллекция='{collection_name}'"
        )

        start_time = time.perf_counter()

        for start_idx in range(0, total_chunks, bs):
            end_idx = min(start_idx + bs, total_chunks)
            batch = chunks[start_idx:end_idx]

            batch_ids = [c["chunk_id"] for c in batch]
            batch_texts = [c["text"] for c in batch]
            batch_metadatas = [self._sanitize_metadata(c.get("metadata", {})) for c in batch]

            # Векторизация батча
            batch_embeddings = embedder.encode_documents(batch_texts, batch_size=bs)

            # Сохранение/обновление в ChromaDB
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts,
            )

            processed = end_idx
            percent = (processed / total_chunks) * 100
            elapsed = time.perf_counter() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"Обработано [{processed}/{total_chunks}] ({percent:.1f}%) "
                f"| Скорость: {rate:.1f} чанков/сек"
            )

        total_elapsed = time.perf_counter() - start_time
        chunks_per_sec = total_chunks / total_elapsed if total_elapsed > 0 else 0.0

        stats = {
            "collection_name": collection_name,
            "model_alias": model_alias,
            "model_name": embedder.model_name,
            "dimension": embedder.dimension,
            "total_chunks": total_chunks,
            "collection_count": collection.count(),
            "elapsed_seconds": round(total_elapsed, 2),
            "chunks_per_second": round(chunks_per_sec, 2),
            "persist_directory": str(self.persist_directory),
        }

        logger.info(
            f"Индексация завершена успешно! "
            f"Всего: {total_chunks} чанков за {total_elapsed:.2f} сек "
            f"({chunks_per_sec:.1f} чанков/сек). В коллекции: {collection.count()} записей."
        )

        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Векторная индексация чанков в ChromaDB")
    parser.add_argument(
        "--file",
        type=str,
        default="data/processed/chunks_by_article.json",
        help="Путь к файлу чанков (JSON)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rubert-tiny2",
        help="Алиас или имя модели эмбеддингов (по умолчанию: rubert-tiny2)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Пользовательское имя коллекции (по умолчанию: {file_stem}_{model_alias})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Размер батча для векторизации",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Очистить коллекцию перед индексацией",
    )
    args = parser.parse_args()

    indexer = VectorIndexer()
    indexer.index_chunks(
        chunks_file_path=args.file,
        model_alias=args.model,
        collection_name=args.collection,
        batch_size=args.batch_size,
        reset_collection=args.reset,
    )


if __name__ == "__main__":
    main()
