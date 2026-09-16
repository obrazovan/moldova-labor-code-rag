"""
Единый конвейер поиска, фильтрации и переранжирования документов.
Связывает цепочку: Query -> Retriever (Top-20) -> Filters -> Reranker (Top-5).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Настройка UTF-8 вывода для Windows консолей
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.retrieval.retriever import VectorRetriever, DEFAULT_RETRIEVAL_CONFIG_PATH
from src.retrieval.filters import ContextFilter
from src.reranking.reranker import DocumentReranker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SearchPipeline:
    """
    Класс конвейера поиска: векторный поиск -> фильтрация контекста -> FlashRank реранкинг.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        collection_name: str = "chunks_by_article_rubert-tiny2",
        embedder_model: str = "rubert-tiny2",
        reranker_model: str | None = None,
    ) -> None:
        """
        Инициализация полного поискового конвейера.
        :param config_path: Путь к файлу configs/retrieval.json.
        :param collection_name: Имя коллекции векторов ChromaDB.
        :param embedder_model: Модель эмбеддингов для векторизации запросов.
        :param reranker_model: Модель реранкера (если None, берется из config).
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_RETRIEVAL_CONFIG_PATH
        self.config = self._load_config(self.config_path)

        self.retriever_top_k = self.config.get("retriever_top_k", 20)
        self.reranker_top_n = self.config.get("reranker_top_n", 5)

        logger.info("Инициализация компонентов SearchPipeline...")
        self.retriever = VectorRetriever(
            collection_name=collection_name,
            model_name_or_alias=embedder_model,
            retrieval_config_path=self.config_path,
        )
        self.filter = ContextFilter()
        self.reranker = DocumentReranker(
            model_name=reranker_model,
            config_path=self.config_path,
        )

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить конфиг {path}: {e}")
        return {}

    def search(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """
        Выполнение полного цикла поиска:
        Query -> VectorRetriever -> ContextFilter -> DocumentReranker.
        :param query: Поисковый запрос.
        :param top_k: Число кандидатов первичного поиска (если None, из конфига).
        :param top_n: Число финальных документов после реранкинга (если None, из конфига).
        :return: Словарь с результатами всех этапов:
                 {
                   'query': str,
                   'raw_retrieved': list[dict],
                   'filtered': list[dict],
                   'reranked': list[dict],
                   'stats': dict
                 }
        """
        k = top_k if top_k is not None else self.retriever_top_k
        n = top_n if top_n is not None else self.reranker_top_n

        logger.info(f"Запуск поиска по запросу: '{query}' (top_k={k}, top_n={n})")

        # 1. Первичный векторный поиск
        raw_candidates = self.retriever.retrieve(query=query, top_k=k)
        count_raw = len(raw_candidates)

        # 2. Фильтрация контекста
        filtered_candidates = self.filter.apply_all(raw_candidates, config=self.config)
        count_filtered = len(filtered_candidates)

        # 3. Реранкинг FlashRank
        final_documents = self.reranker.rerank(
            query=query,
            documents=filtered_candidates,
            top_n=n,
        )
        count_final = len(final_documents)

        logger.info(
            f"Поиск завершен: извлечено {count_raw} -> отфильтровано {count_filtered} -> отобрано в топ-{count_final}"
        )

        return {
            "query": query,
            "raw_retrieved": raw_candidates,
            "filtered": filtered_candidates,
            "reranked": final_documents,
            "stats": {
                "raw_count": count_raw,
                "filtered_count": count_filtered,
                "final_count": count_final,
                "retriever_top_k": k,
                "reranker_top_n": n,
                "similarity_threshold": self.config.get("similarity_threshold", 0.35),
                "min_chunk_chars": self.config.get("min_chunk_chars", 30),
            },
        }


def format_doc_card(rank: int, doc: dict[str, Any], score_key: str, score_label: str) -> str:
    """Форматирует карточку документа для наглядного вывода в консоль."""
    metadata = doc.get("metadata", {})
    art_num = metadata.get("article_number", "—")
    title = metadata.get("title", "Без заголовка")
    score_val = doc.get(score_key, 0.0)
    sim_score = doc.get("similarity_score", 0.0)
    chunk_id = doc.get("id", "—")

    # Превью текста (первые 200 символов, без лишних переносов строк)
    text = doc.get("text", "").replace("\n", " ").strip()
    preview = (text[:220] + "...") if len(text) > 220 else text

    lines = [
        f"  [{rank}] Статья {art_num}: «{title}» (ID: {chunk_id})",
        f"      Балл: {score_label} = {score_val:.4f} (Vector Sim: {sim_score:.4f})",
        f"      Текст: {preview}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Демонстрационный запуск поискового конвейера (Retriever -> Filters -> Reranker)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Какова продолжительность ежегодного оплачиваемого отпуска?",
        help="Тестовый поисковый запрос (по умолчанию: продолжительность отпуска)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Количество кандидатов первичного поиска",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Количество документов после реранкинга",
    )
    args = parser.parse_args()

    pipeline = SearchPipeline()
    result = pipeline.search(query=args.query, top_k=args.top_k, top_n=args.top_n)

    raw_docs = result["raw_retrieved"]
    reranked_docs = result["reranked"]
    stats = result["stats"]

    divider = "=" * 80
    sub_divider = "-" * 80

    print("\n" + divider)
    print("ДЕМОНСТРАЦИЯ ПОИСКОВОГО КОНВЕЙЕРА (ЭТАП 4: RETRIEVER -> FILTERS -> RERANKER)")
    print(divider)
    print(f"Поисковый запрос: \"{result['query']}\"")
    print(
        f"Параметры: Top-K Retriever = {stats['retriever_top_k']} | "
        f"Порог сходства >= {stats['similarity_threshold']} | "
        f"Мин. символов = {stats['min_chunk_chars']} | "
        f"Top-N Reranker = {stats['reranker_top_n']}"
    )
    print(
        f"Воронка документов: Найдено в ChromaDB: {stats['raw_count']} -> "
        f"После фильтров: {stats['filtered_count']} -> "
        f"В топе реранкера: {stats['final_count']}"
    )

    # 1. Топ-3 до реранкинга (из Retriever)
    print("\n" + sub_divider)
    print("1. ТОП-3 ДО РЕРАНКИНГА (Первичная векторная выдача VectorRetriever):")
    print(sub_divider)
    top3_raw = raw_docs[:3]
    if not top3_raw:
        print("  Документы не найдены.")
    else:
        for idx, doc in enumerate(top3_raw, 1):
            print(format_doc_card(idx, doc, "similarity_score", "Similarity Score"))

    # 2. Топ-3 после реранкинга (из Reranker)
    print("\n" + sub_divider)
    print("2. ТОП-3 ПОСЛЕ РЕРАНКИНГА (Выдача Cross-Encoder FlashRank Reranker):")
    print(sub_divider)
    top3_reranked = reranked_docs[:3]
    if not top3_reranked:
        print("  Документы не найдены.")
    else:
        for idx, doc in enumerate(top3_reranked, 1):
            print(format_doc_card(idx, doc, "rerank_score", "FlashRank Score"))

    # 3. Анализ различий
    print("\n" + sub_divider)
    print("3. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ВЛИЯНИЯ РЕРАНКИНГА:")
    print(sub_divider)
    if top3_raw and top3_reranked:
        raw_top1_art = top3_raw[0].get("metadata", {}).get("article_number")
        rerank_top1_art = top3_reranked[0].get("metadata", {}).get("article_number")

        print(f"  • Лидер векторного поиска:    Статья {raw_top1_art} (Similarity = {top3_raw[0].get('similarity_score', 0):.4f})")
        print(f"  • Лидер после FlashRank:     Статья {rerank_top1_art} (FlashRank Score = {top3_reranked[0].get('rerank_score', 0):.4f})")

        # Позиция лидера реранкера в исходной выдаче
        rerank_top1_id = top3_reranked[0].get("id")
        orig_pos = next((i + 1 for i, d in enumerate(raw_docs) if d.get("id") == rerank_top1_id), None)
        if orig_pos:
            print(f"  • Позиция целевой статьи {rerank_top1_art} в первичном поиске: #{orig_pos} -> поднялась на #1")

    print(divider + "\n")


if __name__ == "__main__":
    main()
