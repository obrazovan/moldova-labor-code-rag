"""
Модуль векторного представления текста (embeddings).
Обертка над SentenceTransformer с поддержкой инференса на CPU и специфических префиксов (e5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "embeddings.json"


class EmbeddingModel:
    """
    Универсальная обертка над моделями SentenceTransformer для инференса на CPU.
    """

    def __init__(
        self,
        model_name_or_alias: str = "rubert-tiny2",
        config_path: str | Path | None = None,
        batch_size: int = 32,
    ) -> None:
        """
        Инициализация модели.
        :param model_name_or_alias: Алиас из configs/embeddings.json (например, rubert-tiny2, multilingual-e5-small)
                                     или валидный идентификатор HuggingFace.
        :param config_path: Путь к конфигурационному файлу (по умолчанию configs/embeddings.json).
        :param batch_size: Размер батча по умолчанию.
        """
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config = self._load_config(self._config_path)

        self._model_alias = model_name_or_alias
        self._batch_size = batch_size

        models_cfg = self._config.get("models", {})
        if model_name_or_alias in models_cfg:
            entry = models_cfg[model_name_or_alias]
            self._model_full_name: str = entry.get("name", model_name_or_alias)
            self._dimension: int | None = entry.get("dimensions")
        else:
            self._model_full_name = model_name_or_alias
            self._dimension = None

        if "batch_size" in self._config:
            self._batch_size = self._config["batch_size"]

        # Принудительная инициализация на CPU
        self.model = SentenceTransformer(self._model_full_name, device="cpu")

        if self._dimension is None:
            if hasattr(self.model, "get_embedding_dimension"):
                self._dimension = int(self.model.get_embedding_dimension())
            else:
                self._dimension = int(self.model.get_sentence_embedding_dimension())

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @property
    def dimension(self) -> int:
        """Размерность вектора эмбеддинга."""
        if self._dimension is not None:
            return self._dimension
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        """Имя модели (HuggingFace identifier)."""
        return self._model_full_name

    @property
    def model_alias(self) -> str:
        """Алиас модели (например, rubert-tiny2)."""
        return self._model_alias

    def encode_documents(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """
        Векторизация списка документов/чанков.
        :param texts: Список строк для кодирования.
        :param batch_size: Размер батча при векторизации (если None, берется из конфига).
        :return: Список векторов с нормализованной длиной (L2 norm) для косинусного сходства.
        """
        if not texts:
            return []

        bs = batch_size or self._batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_query(self, text: str) -> list[float]:
        """
        Векторизация одиночного поискового запроса.
        Для моделей семейств e5 добавляет префикс 'query: '.
        Для rubert-tiny2 и других моделей возвращает эмбеддинг оригинального текста.
        :param text: Текст запроса.
        :return: Нормализованный вектор запроса (L2 norm).
        """
        formatted_text = text
        is_e5 = (
            "e5" in self._model_alias.lower()
            or "e5" in self._model_full_name.lower()
        )
        if is_e5 and not formatted_text.startswith("query: "):
            formatted_text = f"query: {formatted_text}"

        embedding = self.model.encode(
            formatted_text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
