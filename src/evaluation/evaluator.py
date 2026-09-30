"""
Модуль проведения тестирования поискового конвейера и сквозной RAG-системы.
Реализует класс RAGEvaluator для замера метрик IR (Hit Rate, MRR, Precision, Recall)
и оценки качества генерации ответов (защита от галлюцинаций, покрытие ключевых слов, источники).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Настройка UTF-8 для вывода в Windows консоль
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.evaluation.metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_precision_recall_at_k,
    evaluate_hallucination_refusal,
    compute_aggregate_metrics,
)
from src.retrieval.search_pipeline import SearchPipeline
from src.generation.rag_service import RAGService

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "test_dataset.json"


class RAGEvaluator:
    """
    Класс для автоматизированного тестирования компонентов RAG:
    - Поисковый конвейер (Retriever + Filter + Reranker)
    - Сквозная генерация и анти-галлюцинации (End-to-End RAG)
    """

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        """
        Инициализация оценщика.
        :param dataset_path: Путь к файлу test_dataset.json.
        """
        self.dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH

    def load_dataset(self, dataset_path: str | Path | None = None) -> list[dict[str, Any]]:
        """Загружает тестовый датасет из JSON файла."""
        path = Path(dataset_path) if dataset_path else self.dataset_path
        if not path.exists():
            raise FileNotFoundError(f"Тестовый датасет не найден: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Ожидался список объектов в {path}, получено: {type(data)}")

        return data

    @staticmethod
    def extract_article_numbers_from_docs(docs: list[dict[str, Any]]) -> list[str]:
        """
        Извлекает уникальные упорядоченные номера статей из списка найденных фрагментов.
        Сохраняет исходный порядок релевантности первого появления статьи.
        """
        articles: list[str] = []
        seen: set[str] = set()

        for doc in docs:
            meta = doc.get("metadata", {})
            art_num = meta.get("article_number")
            if art_num is not None:
                art_str = str(art_num).strip()
                if art_str and art_str not in seen:
                    seen.add(art_str)
                    articles.append(art_str)

        return articles

    def evaluate_retrieval(
        self,
        dataset_path: str | Path | None = None,
        search_pipeline: SearchPipeline | None = None,
        k_values: list[int] = [3, 5, 10, 20],
        use_reranked: bool = True,
        similarity_threshold: float | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Прогоняет все answerable вопросы через поиск, извлекает номера найденных статей
        и считает усредненные Hit Rate@K, MRR@K, Precision@K, Recall@K.

        :param dataset_path: Путь к test_dataset.json.
        :param search_pipeline: Экземпляр SearchPipeline (если None, создается новый).
        :param k_values: Список значений глубины поиска K.
        :param use_reranked: Оценивать выдачу после реранкера (True) или сырую выдачу retriever (False).
        :param similarity_threshold: Опциональный порог сходства для переопределения.
        :param verbose: Логировать ли каждый отдельный запрос.
        :return: Словарь с агрегированными и подробными результатами.
        """
        dataset = self.load_dataset(dataset_path)
        pipeline = search_pipeline or SearchPipeline()
        max_k = max(k_values) if k_values else 20

        # Фильтруем вопросы: в retrieval оцениваются запросы с известными целевыми статьями (answerable)
        eval_items = [item for item in dataset if item.get("is_answerable", True)]

        eval_records: list[dict[str, Any]] = []
        start_time = time.perf_counter()

        for idx, item in enumerate(eval_items, start=1):
            qid = item.get("id", idx)
            query = item["question"]
            target_articles = [str(a).strip() for a in item.get("target_articles", [])]
            category = item.get("category", "general")

            # Выполняем поиск
            search_res = pipeline.search(
                query=query,
                top_k=max_k,
                top_n=max_k,
                similarity_threshold=similarity_threshold,
                verbose=False,
            )

            if use_reranked:
                retrieved_docs = search_res.get("reranked", [])
                # Если реранкер вернул пустой список из-за строгого фильтра, страхуемся
                if not retrieved_docs and not search_res.get("filtered"):
                    retrieved_docs = search_res.get("raw_retrieved", [])
            else:
                if similarity_threshold is not None:
                    retrieved_docs = search_res.get("filtered", [])
                else:
                    retrieved_docs = search_res.get("raw_retrieved", [])

            retrieved_articles = self.extract_article_numbers_from_docs(retrieved_docs)

            # Расчет индивидуальных метрик
            rr = calculate_mrr(retrieved_articles, target_articles)
            hits_by_k = {
                f"hit@{k}": calculate_hit_rate(retrieved_articles, target_articles, k)
                for k in k_values
            }

            record = {
                "id": qid,
                "question": query,
                "category": category,
                "target_articles": target_articles,
                "retrieved_articles": retrieved_articles[:max_k],
                "mrr": round(rr, 4),
                **hits_by_k,
            }
            eval_records.append(record)

            if verbose:
                logger.info(
                    f"[{idx}/{len(eval_items)}] Q{qid} ({category}): "
                    f"Target={target_articles} | Found={retrieved_articles[:5]} | RR={rr:.2f}"
                )

        elapsed = time.perf_counter() - start_time

        # Агрегация общих метрик
        overall_summary = compute_aggregate_metrics(eval_records, k_values=k_values)
        overall_summary["elapsed_seconds"] = round(elapsed, 2)
        overall_summary["avg_latency_ms"] = round((elapsed / len(eval_items)) * 1000, 2) if eval_items else 0.0

        # Агрегация метрик в разрезе категорий
        categories = sorted(list({rec["category"] for rec in eval_records}))
        by_category: dict[str, dict[str, Any]] = {}
        for cat in categories:
            cat_records = [r for r in eval_records if r["category"] == cat]
            by_category[cat] = compute_aggregate_metrics(cat_records, k_values=k_values)

        return {
            "total_queries_evaluated": len(eval_records),
            "k_values": k_values,
            "overall": overall_summary,
            "by_category": by_category,
            "detailed_results": eval_records,
        }

    def evaluate_end_to_end(
        self,
        dataset_path: str | Path | None = None,
        rag_service: RAGService | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Оценивает сквозную генерацию ответов RAG:
        1. Защита от галлюцинаций (процент отказов на out-of-domain unanswerable запросах).
        2. Корректность ответов на answerable запросы (наличие ключевых слов и отсутствие ложных отказов).
        3. Точность атрибуции нормативных источников (указание статей ТК РМ).

        :param dataset_path: Путь к test_dataset.json.
        :param rag_service: Экземпляр RAGService (если None, создается новый).
        :param verbose: Логировать ли каждый шаг генерации.
        :return: Словарь с агрегированными и подробными результатами тестирования генерации.
        """
        dataset = self.load_dataset(dataset_path)
        service = rag_service or RAGService()

        total_queries = len(dataset)
        unanswerable_items = [q for q in dataset if not q.get("is_answerable", True)]
        answerable_items = [q for q in dataset if q.get("is_answerable", True)]

        results: list[dict[str, Any]] = []
        start_time = time.perf_counter()

        correct_refusal_count = 0
        hallucination_count = 0
        valid_response_count = 0
        all_keywords_count = 0
        partial_keywords_count = 0
        correct_source_attribution_count = 0

        for idx, item in enumerate(dataset, start=1):
            qid = item.get("id", idx)
            query = item["question"]
            is_answerable = item.get("is_answerable", True)
            target_articles = [str(a).strip() for a in item.get("target_articles", [])]
            expected_keywords = item.get("expected_answer_keywords", [])
            category = item.get("category", "general")

            q_start = time.perf_counter()
            response = service.answer(query=query)
            q_elapsed = time.perf_counter() - q_start

            answer_text = response.get("answer", "")
            sources = response.get("sources", [])
            retrieved_docs = response.get("retrieved_docs", [])
            retrieved_articles = self.extract_article_numbers_from_docs(retrieved_docs)

            # Проверка поведения (отказ vs содержательный ответ)
            is_refusal_behavior_correct = evaluate_hallucination_refusal(answer_text, is_answerable)

            # Анализ ключевых слов
            matched_keywords: list[str] = []
            if is_answerable and expected_keywords:
                answer_lower = answer_text.lower()
                for kw in expected_keywords:
                    if kw.lower() in answer_lower:
                        matched_keywords.append(kw)

            # Анализ атрибуции источников
            source_hit = False
            if is_answerable and target_articles:
                combined_source_text = " ".join(sources).lower() + " " + answer_text.lower()
                for target_art in target_articles:
                    target_str = f"статья {target_art.lower()}"
                    target_num_str = f" {target_art.lower()}"
                    if target_str in combined_source_text or target_num_str in combined_source_text:
                        source_hit = True
                        break

            if not is_answerable:
                if is_refusal_behavior_correct:
                    correct_refusal_count += 1
                else:
                    hallucination_count += 1
            else:
                if is_refusal_behavior_correct:
                    valid_response_count += 1
                if len(matched_keywords) == len(expected_keywords) and len(expected_keywords) > 0:
                    all_keywords_count += 1
                if len(matched_keywords) > 0:
                    partial_keywords_count += 1
                if source_hit:
                    correct_source_attribution_count += 1

            record = {
                "id": qid,
                "question": query,
                "category": category,
                "is_answerable": is_answerable,
                "target_articles": target_articles,
                "answer": answer_text,
                "sources": sources,
                "retrieved_articles": retrieved_articles,
                "correct_behavior": is_refusal_behavior_correct,
                "matched_keywords": matched_keywords,
                "expected_keywords": expected_keywords,
                "source_attributed_correctly": source_hit,
                "latency_sec": round(q_elapsed, 2),
            }
            results.append(record)

            if verbose:
                logger.info(
                    f"[{idx}/{total_queries}] Q{qid} ({category}): "
                    f"Correct={is_refusal_behavior_correct} | Latency={q_elapsed:.2f}s"
                )

        total_elapsed = time.perf_counter() - start_time

        num_unanswerable = len(unanswerable_items)
        num_answerable = len(answerable_items)

        summary = {
            "total_queries": total_queries,
            "answerable_count": num_answerable,
            "unanswerable_count": num_unanswerable,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "avg_latency_seconds": round(total_elapsed / total_queries, 2) if total_queries else 0.0,
            # Метрики галлюцинаций
            "unanswerable_refusal_rate": round(correct_refusal_count / num_unanswerable, 4) if num_unanswerable else 1.0,
            "hallucination_rate": round(hallucination_count / num_unanswerable, 4) if num_unanswerable else 0.0,
            # Метрики полноты ответов
            "answerable_response_rate": round(valid_response_count / num_answerable, 4) if num_answerable else 0.0,
            "full_keyword_match_rate": round(all_keywords_count / num_answerable, 4) if num_answerable else 0.0,
            "partial_keyword_match_rate": round(partial_keywords_count / num_answerable, 4) if num_answerable else 0.0,
            # Метрика атрибуции источников
            "source_attribution_accuracy": round(correct_source_attribution_count / num_answerable, 4) if num_answerable else 0.0,
        }

        return {
            "summary": summary,
            "detailed_results": results,
        }
