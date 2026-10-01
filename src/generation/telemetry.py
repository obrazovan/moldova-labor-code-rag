"""
Модуль телеметрии и наблюдаемости (Observability, Langfuse Tracing, Latency, Tokens & Cost Tracking).
Обеспечивает сквозную трассировку RAG-пайплайна, замер задержек поиска и генерации,
учет расхода токенов и расчет оценочной стоимости запросов Google Gemini.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv

# Настройка UTF-8 вывода для Windows консолей
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)


class _NoOpLangfuse:
    """Заглушка для безопасного вызова методов Langfuse в No-Op режиме без исключений."""

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_trace_url(self, *args: Any, **kwargs: Any) -> str | None:
        return None


class TelemetryTracker:
    """
    Класс для наблюдаемости RAG: трассировка в Langfuse,
    подсчет расхода токенов, замер задержек поиска/генерации и расчет стоимости.
    """

    # Тарифные ставки за 1 000 000 токенов (USD) для линейки Google Gemini Flash
    # Стандартные тарифы: $0.075 за 1M входных (prompt), $0.30 за 1M выходных (output)
    PRICING_PER_1M: dict[str, dict[str, float]] = {
        "default": {"prompt": 0.075, "output": 0.30},
        "gemini-2.5-flash": {"prompt": 0.075, "output": 0.30},
        "gemini-3.5-flash": {"prompt": 0.075, "output": 0.30},
        "gemini-3.5-flash-lite": {"prompt": 0.075, "output": 0.30},
        "gemini-3.8-flash": {"prompt": 0.075, "output": 0.30},
        "gemini-flash-latest": {"prompt": 0.075, "output": 0.30},
    }

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        """
        Инициализация трекера телеметрии.
        Если ключи Langfuse не заданы или некорректны, переходит в тихий локальный режим (No-Op).
        """
        load_dotenv()

        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        self.is_active = False
        self.client = None
        self.langfuse: Any = _NoOpLangfuse()

        # Проверка валидности ключей (отсекаем плейсхолдеры и пустые значения)
        if (
            self.public_key
            and self.secret_key
            and not self.public_key.startswith("your_")
            and not self.public_key.startswith("pk-lf-...")
            and not self.secret_key.startswith("your_")
            and not self.secret_key.startswith("sk-lf-...")
        ):
            try:
                from langfuse import Langfuse

                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                self.langfuse = self.client
                self.is_active = True
                logger.info(f"Langfuse успешно подключен (host: {self.host})")
            except Exception as exc:
                logger.warning(
                    f"Не удалось инициализировать клиент Langfuse ({exc}). "
                    "TelemetryTracker переведен в тихий локальный режим (No-Op)."
                )
                self.is_active = False
                self.client = None
                self.langfuse = _NoOpLangfuse()
        else:
            logger.info(
                "Ключи Langfuse не настроены в .env. TelemetryTracker работает в тихом локальном режиме (No-Op)."
            )
            self.langfuse = _NoOpLangfuse()

    def estimate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Расчет оценочной стоимости выполнения запроса в долларах США.
        Для линейки Flash: $0.075 за 1M входных токенов, $0.30 за 1M выходных токенов.

        :param model_name: Имя языковой модели.
        :param input_tokens: Количество токенов контекста и промпта.
        :param output_tokens: Количество токенов сгенерированного ответа.
        :return: Оценочная стоимость в USD.
        """
        model_key = (model_name or "default").lower().strip()
        rates = self.PRICING_PER_1M.get(model_key, self.PRICING_PER_1M["default"])

        input_cost = (max(0, input_tokens) / 1_000_000.0) * rates["prompt"]
        output_cost = (max(0, output_tokens) / 1_000_000.0) * rates["output"]

        return input_cost + output_cost

    def log_rag_execution(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        final_answer: str,
        search_latency_ms: float,
        llm_latency_ms: float,
        usage_metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        context_text: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Логирование полного выполнения RAG в Langfuse со вложенными спанами retrieval и generation.

        :param query: Исходный вопрос пользователя.
        :param retrieved_docs: Документы, извлеченные поисковым пайплайном.
        :param final_answer: Итоговый сгенерированный ответ LLM.
        :param search_latency_ms: Время поиска в миллисекундах.
        :param llm_latency_ms: Время генерации ответа в миллисекундах.
        :param usage_metadata: Словарь статистики токенов (prompt_token_count, candidates_token_count).
        :param system_prompt: Системный промпт (инструкция).
        :param context_text: Сформированный контекст статей.
        :param model_name: Имя вызванной модели.
        :return: Словарь с метриками и URL трейса.
        """
        meta = usage_metadata or {}
        input_tokens = int(meta.get("prompt_token_count", 0) or 0)
        output_tokens = int(meta.get("candidates_token_count", 0) or 0)
        total_tokens = int(meta.get("total_token_count", input_tokens + output_tokens) or 0)

        effective_model = model_name or "gemini-3.5-flash-lite"
        cost = self.estimate_cost(effective_model, input_tokens, output_tokens)
        total_latency_ms = search_latency_ms + llm_latency_ms

        trace_id = None
        trace_url = None

        if self.is_active and self.client is not None:
            try:
                # 1. Корневой спан RAG (тип chain)
                root = self.client.start_observation(
                    name="rag_pipeline",
                    as_type="chain",
                    input={"query": query},
                    metadata={
                        "model": effective_model,
                        "search_latency_ms": round(search_latency_ms, 2),
                        "llm_latency_ms": round(llm_latency_ms, 2),
                        "total_latency_ms": round(total_latency_ms, 2),
                        "estimated_cost_usd": round(cost, 6),
                    },
                )
                trace_id = getattr(root, "trace_id", None)

                # 2. Вложенный спан retrieval (тип retriever)
                docs_summary = [
                    {
                        "id": doc.get("id"),
                        "article_number": doc.get("metadata", {}).get("article_number"),
                        "title": doc.get("metadata", {}).get("title"),
                        "similarity_score": doc.get("similarity_score"),
                        "rerank_score": doc.get("rerank_score"),
                    }
                    for doc in (retrieved_docs or [])
                ]

                ret_span = root.start_observation(
                    name="retrieval",
                    as_type="retriever",
                    input={"query": query},
                    output=docs_summary,
                    metadata={
                        "candidates_count": len(retrieved_docs or []),
                        "latency_ms": round(search_latency_ms, 2),
                    },
                )
                ret_span.end()

                # 3. Вложенный спан generation (тип generation)
                gen_span = root.start_observation(
                    name="generation",
                    as_type="generation",
                    input={
                        "system_prompt": system_prompt or "",
                        "context": context_text or "",
                        "query": query,
                    },
                    output=final_answer,
                    model=effective_model,
                    usage_details={
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": total_tokens,
                    },
                    cost_details={
                        "total": cost,
                    },
                    metadata={
                        "latency_ms": round(llm_latency_ms, 2),
                    },
                )
                gen_span.end()

                # 4. Завершение корневого спана
                root.update(output={"answer": final_answer})
                root.end()

                # 5. Получение ссылки на трейс
                if trace_id:
                    try:
                        trace_url = self.langfuse.get_trace_url(trace_id=trace_id)
                    except Exception:
                        trace_url = None
                    if not trace_url:
                        trace_url = f"{self.host.rstrip('/')}/trace/{trace_id}"

                # Принудительный сброс буфера перед выходом из метода,
                # чтобы фоновый поток успевал отправить телеметрию в cloud.langfuse.com
                if self.langfuse is not None:
                    try:
                        self.langfuse.flush()
                    except Exception as e:
                        logger.warning(f"Ошибка при вызове self.langfuse.flush(): {e}")

            except Exception as e:
                logger.warning(f"Ошибка при отправке трейса в Langfuse: {e}")

        # Если Langfuse не активен, формируем статус для плашки
        if not trace_url:
            if not self.is_active:
                trace_url = "Не настроен (No-Op mode)"
            else:
                trace_url = "Доступен в дашборде Langfuse"

        return {
            "trace_id": trace_id,
            "trace_url": trace_url,
            "is_logged": bool(trace_id and self.is_active),
            "search_latency_ms": search_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "total_latency_ms": total_latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "model_name": effective_model,
        }

    def flush(self) -> None:
        """Принудительный сброс буфера событий в Langfuse."""
        if self.langfuse is not None:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.warning(f"Ошибка при вызове self.langfuse.flush(): {e}")

    def shutdown(self) -> None:
        """Корректное закрытие клиента Langfuse с гарантированной отправкой телеметрии."""
        if self.langfuse is not None:
            try:
                self.langfuse.shutdown()
            except Exception as e:
                logger.warning(f"Ошибка при вызове self.langfuse.shutdown(): {e}")

    def format_metrics_badge(self, metrics: dict[str, Any]) -> str:
        """
        Форматирует сводную плашку метрик в консоли согласно ТЗ:
        [МЕТРИКИ ЗАПРОСА]:
        • Задержка: 1.45 сек (Поиск: 0.15 сек | LLM: 1.30 сек)
        • Токены: 1,340 (Контекст: 1,180 | Ответ: 160)
        • Оценочная стоимость: $0.000137
        • Langfuse Trace: https://cloud.langfuse.com/project/.../traces/...
        """
        total_sec = metrics.get("total_latency_ms", 0.0) / 1000.0
        search_sec = metrics.get("search_latency_ms", 0.0) / 1000.0
        llm_sec = metrics.get("llm_latency_ms", 0.0) / 1000.0

        tokens_total = metrics.get("total_tokens", 0)
        tokens_in = metrics.get("input_tokens", 0)
        tokens_out = metrics.get("output_tokens", 0)

        cost = metrics.get("estimated_cost_usd", 0.0)
        trace_url = metrics.get("trace_url", "Не настроен (No-Op mode)")

        lines = [
            "[МЕТРИКИ ЗАПРОСА]:",
            f"• Задержка: {total_sec:.2f} сек (Поиск: {search_sec:.2f} сек | LLM: {llm_sec:.2f} сек)",
            f"• Токены: {tokens_total:,} (Контекст: {tokens_in:,} | Ответ: {tokens_out:,})",
            f"• Оценочная стоимость: ${cost:.6f}",
            f"• Langfuse Trace: {trace_url}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    tracker = TelemetryTracker()
    print("Проверка TelemetryTracker...")
    cost_test = tracker.estimate_cost("gemini-3.5-flash-lite", 1180, 160)
    print(f"Тест расчета стоимости (1180 in, 160 out): ${cost_test:.6f} (Ожидалось: ~$0.000137)")

    mock_metrics = tracker.log_rag_execution(
        query="Какова продолжительность отпуска?",
        retrieved_docs=[{"id": "doc1", "metadata": {"article_number": "113", "title": "Отпуск"}, "similarity_score": 0.88}],
        final_answer="Продолжительность отпуска составляет 28 дней.",
        search_latency_ms=150.0,
        llm_latency_ms=1300.0,
        usage_metadata={"prompt_token_count": 1180, "candidates_token_count": 160, "total_token_count": 1340},
    )
    print("\n" + tracker.format_metrics_badge(mock_metrics))
