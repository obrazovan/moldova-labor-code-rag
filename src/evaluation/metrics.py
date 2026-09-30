"""
Модуль расчета метрик качества информационного поиска (Information Retrieval)
и оценки работы RAG-системы по Трудовому кодексу Республики Молдова.

Реализует стандартные метрики ранжирования и проверки галлюцинаций:
- Hit Rate@K (коэффициент попадания целевой статьи в Top-K)
- MRR (Mean Reciprocal Rank — обратный ранг первого релевантного документа)
- Precision@K и Recall@K (точность и полнота на глубине K)
- Hallucination / Refusal Evaluation (проверка корректности отказа на out-of-domain запросы)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Маркеры стандартного отказа при недостатке информации или out-of-domain запросах
STANDARD_REFUSAL_MARKERS: list[str] = [
    "информации недостаточно",
    "нет информации",
    "не содержит информации",
    "не содержит сведений",
    "не указано в предоставленных",
    "не регулирует данный вопрос",
    "не относится к трудовому кодексу",
    "не содержится в предоставленном контексте",
    "в предоставленных статьях трудового кодекса",
    "не представляется возможным ответить",
]


def _normalize_article(article_str: Any) -> str:
    """Нормализует номер статьи для точного сопоставления (удаляет пробелы, спецсимволы)."""
    if article_str is None:
        return ""
    return str(article_str).strip().lower()


def calculate_hit_rate(
    retrieved_articles: Sequence[str],
    target_articles: Sequence[str],
    k: int,
) -> float:
    """
    Вычисляет Hit Rate@K: возвращает 1.0, если хотя бы одна целевая статья
    присутствует среди первых K извлеченных статей, иначе 0.0.

    :param retrieved_articles: Упорядоченный список номеров найденных статей.
    :param target_articles: Список эталонных номеров статей.
    :param k: Глубина проверки (Top-K).
    :return: 1.0 (попадание) или 0.0 (промах).
    """
    if k <= 0 or not target_articles or not retrieved_articles:
        return 0.0

    target_set = {_normalize_article(art) for art in target_articles if _normalize_article(art)}
    if not target_set:
        return 0.0

    top_k_retrieved = [_normalize_article(art) for art in retrieved_articles[:k]]
    for art in top_k_retrieved:
        if art in target_set:
            return 1.0

    return 0.0


def calculate_mrr(
    retrieved_articles: Sequence[str],
    target_articles: Sequence[str],
) -> float:
    """
    Вычисляет Reciprocal Rank (RR): возвращает 1 / rank первого вхождения
    любой из целевых статей в списке извлеченных документов (1-based rank),
    или 0.0, если ни одна целевая статья не найдена.

    :param retrieved_articles: Упорядоченный список номеров найденных статей.
    :param target_articles: Список эталонных номеров статей.
    :return: 1.0 / rank (от 0.0 до 1.0).
    """
    if not target_articles or not retrieved_articles:
        return 0.0

    target_set = {_normalize_article(art) for art in target_articles if _normalize_article(art)}
    if not target_set:
        return 0.0

    for rank, art in enumerate(retrieved_articles, start=1):
        if _normalize_article(art) in target_set:
            return 1.0 / rank

    return 0.0


def calculate_precision_recall_at_k(
    retrieved_articles: Sequence[str],
    target_articles: Sequence[str],
    k: int,
) -> tuple[float, float]:
    """
    Вычисляет Precision@K и Recall@K на уровне уникальных релевантных статей.

    Precision@K = (Количество найденных целевых статей в Top-K) / K
    Recall@K = (Количество найденных целевых статей в Top-K) / (Общее количество целевых статей)

    :param retrieved_articles: Упорядоченный список номеров найденных статей.
    :param target_articles: Список эталонных номеров статей.
    :param k: Глубина проверки (Top-K).
    :return: Кортеж (Precision@K, Recall@K).
    """
    if k <= 0 or not target_articles:
        return 0.0, 0.0

    target_set = {_normalize_article(art) for art in target_articles if _normalize_article(art)}
    if not target_set:
        return 0.0, 0.0

    # Уникальные статьи в первых K позициях
    retrieved_k = {_normalize_article(art) for art in retrieved_articles[:k] if _normalize_article(art)}
    hits = retrieved_k & target_set

    precision = len(hits) / float(k)
    recall = len(hits) / float(len(target_set))

    return precision, recall


def evaluate_hallucination_refusal(answer: str, is_answerable: bool) -> bool:
    """
    Проверяет корректность генерации и защиту от галлюцинаций:
    - Для unanswerable (out-of-domain) запроса (is_answerable=False):
      возвращает True, если модель корректно отказалась отвечать стандартной фразой.
    - Для answerable запроса (is_answerable=True):
      возвращает True, если модель предоставила содержательный ответ (не выдала ложный отказ).

    :param answer: Сгенерированный LLM текст ответа.
    :param is_answerable: Флаг из датасета, можно ли ответить по ТК РМ.
    :return: True, если поведение модели эталонно корректно.
    """
    if not answer or not answer.strip():
        # Пустой ответ на вопрос — ошибка
        return False

    answer_lower = answer.lower()
    has_refusal_marker = any(marker in answer_lower for marker in STANDARD_REFUSAL_MARKERS)

    if not is_answerable:
        # Для нерелевантного вопроса модель ОБЯЗАНА зафиксировать отказ
        return has_refusal_marker
    else:
        # Для релевантного вопроса модель НЕ должна отказываться отвечать
        return not has_refusal_marker


def compute_aggregate_metrics(
    eval_records: list[dict[str, Any]],
    k_values: list[int] = [3, 5, 10, 20],
) -> dict[str, Any]:
    """
    Агрегирует метрики по списку отдельных записей оценки:
    вычисляет средний Hit Rate@K, средний MRR, среднюю точность и полноту.

    :param eval_records: Список словарей с полями:
                         'retrieved_articles', 'target_articles', 'category'.
    :param k_values: Список исследуемых значений K.
    :return: Структурированный словарь с агрегированными показателями.
    """
    if not eval_records:
        return {
            "total_queries": 0,
            "mrr": 0.0,
            **{f"hit_rate@{k}": 0.0 for k in k_values},
            **{f"precision@{k}": 0.0 for k in k_values},
            **{f"recall@{k}": 0.0 for k in k_values},
        }

    total = len(eval_records)
    mrr_sum = sum(
        calculate_mrr(rec["retrieved_articles"], rec["target_articles"])
        for rec in eval_records
    )

    summary: dict[str, Any] = {
        "total_queries": total,
        "mrr": round(mrr_sum / total, 4),
    }

    for k in k_values:
        hit_sum = sum(
            calculate_hit_rate(rec["retrieved_articles"], rec["target_articles"], k=k)
            for rec in eval_records
        )
        p_sum = 0.0
        r_sum = 0.0
        for rec in eval_records:
            p, r = calculate_precision_recall_at_k(rec["retrieved_articles"], rec["target_articles"], k=k)
            p_sum += p
            r_sum += r

        mrr_k_sum = sum(
            calculate_mrr(rec["retrieved_articles"][:k], rec["target_articles"])
            for rec in eval_records
        )
        summary[f"hit_rate@{k}"] = round(hit_sum / total, 4)
        summary[f"mrr@{k}"] = round(mrr_k_sum / total, 4)
        summary[f"precision@{k}"] = round(p_sum / total, 4)
        summary[f"recall@{k}"] = round(r_sum / total, 4)

    return summary
