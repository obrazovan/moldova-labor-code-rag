"""
Сквозной сервис RAG (Retrieval-Augmented Generation) по Трудовому кодексу Республики Молдова.
Оркестрирует цепочку: SearchPipeline -> PromptBuilder -> GeminiClient.
Включает механизмы защиты от галлюцинаций и демонстрационные тесты.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Настройка UTF-8 вывода для Windows консолей
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.retrieval.search_pipeline import SearchPipeline
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import GeminiClient
from src.generation.telemetry import TelemetryTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RAGService:
    """
    Главный сквозной сервис вопросно-ответной системы RAG.
    Объединяет поиск (Retriever + Filter + Reranker), форматирование промпта,
    генерацию LLM и систему телеметрии/наблюдаемости (Langfuse Tracing).
    """

    def __init__(
        self,
        search_pipeline: SearchPipeline | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_client: GeminiClient | None = None,
        telemetry: TelemetryTracker | None = None,
        retrieval_config_path: str | Path | None = None,
        generation_config_path: str | Path | None = None,
    ) -> None:
        """
        Инициализация сквозного RAG-сервиса.

        :param search_pipeline: Экземпляр поискового пайплайна (если None, создается новый).
        :param prompt_builder: Экземпляр строителя промптов (если None, создается новый).
        :param llm_client: Клиент Gemini (если None, инициализируется с дефолтным конфигом).
        :param telemetry: Трекер телеметрии и трассировки Langfuse.
        :param retrieval_config_path: Путь к файлу configs/retrieval.json.
        :param generation_config_path: Путь к файлу configs/generation.json.
        """
        logger.info("Инициализация RAGService...")

        self.search_pipeline = search_pipeline or SearchPipeline(config_path=retrieval_config_path)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client or GeminiClient(config_path=generation_config_path)
        self.telemetry = telemetry or TelemetryTracker()

        logger.info("RAGService успешно инициализирован и готов к обработке запросов.")

    def _extract_sources(self, answer: str, retrieved_docs: list[dict[str, Any]]) -> list[str]:
        """
        Извлекает нормативные источники из ответа LLM или из переданных фрагментов.
        Если в ответе зафиксирован отказ из-за нехватки информации, список источников пуст.

        :param answer: Сгенерированный ответ LLM.
        :param retrieved_docs: Документы, поданные в контекст.
        :return: Список строк-источников вида 'Статья X. Название'.
        """
        # Если модель сообщила о недостаточности информации, источников нет
        if "информации недостаточно" in answer.lower():
            return []

        sources: list[str] = []

        # 1. Попытка распарсить блок 'Источники:' из ответа модели
        if "источники:" in answer.lower():
            idx = answer.lower().rfind("источники:")
            sources_section = answer[idx + len("источники:"):].strip()
            for line in sources_section.splitlines():
                cleaned = line.strip().lstrip("-*•0123456789.) ").strip()
                if cleaned and "статья" in cleaned.lower():
                    sources.append(cleaned)

        # 2. Если в ответе раздел 'Источники:' не выделился построчно,
        # извлекаем уникальные статьи из переданных в контекст документов
        if not sources and retrieved_docs:
            sources = self.prompt_builder.extract_sources_from_docs(retrieved_docs)

        return sources

    def answer(self, query: str) -> dict[str, Any]:
        """
        Сквозное выполнение запроса:
        1. Поиск и реранкинг релевантных статей (SearchPipeline) с замером задержки.
        2. Формирование структурированного контекста (PromptBuilder).
        3. Генерация ответа через LLM с защитой от галлюцинаций (GeminiClient) с замером задержки.
        4. Логирование спанов и трейсов в Langfuse, расчет токенов и стоимости (TelemetryTracker).
        5. Формирование структуры результата, источников и плашки метрик.

        :param query: Вопрос пользователя.
        :return: Словарь с ключами:
                 - 'query': текст вопроса
                 - 'answer': сгенерированный LLM ответ
                 - 'retrieved_docs': список отобранных реранкером документов
                 - 'sources': список нормативных источников
                 - 'metrics': метрики выполнения (задержки, токены, стоимость, трейс)
                 - 'metrics_badge': отформатированная строка для вывода в CLI
        """
        clean_query = query.strip()
        logger.info(f"Обработка запроса пользователя: '{clean_query}'")

        t_total_start = time.perf_counter()

        # 1. Поиск релевантных документов с точным замером времени
        t_search_start = time.perf_counter()
        search_result = self.search_pipeline.search(query=clean_query)
        search_latency_sec = time.perf_counter() - t_search_start
        search_latency_ms = search_latency_sec * 1000.0
        retrieved_docs = search_result.get("reranked", [])

        # 2. Формирование промпта с контекстом
        user_message = self.prompt_builder.build_user_message(
            query=clean_query,
            documents=retrieved_docs,
        )

        # 3. Вызов Gemini LLM с точным замером времени
        t_llm_start = time.perf_counter()
        gen_result = self.llm_client.generate(prompt=user_message)
        llm_latency_sec = time.perf_counter() - t_llm_start
        llm_latency_ms = llm_latency_sec * 1000.0

        generated_answer = gen_result.text
        usage_metadata = getattr(gen_result, "usage_metadata", {})

        total_latency_sec = time.perf_counter() - t_total_start
        total_latency_ms = total_latency_sec * 1000.0

        # 4. Извлечение источников
        sources = self._extract_sources(generated_answer, retrieved_docs)

        # 5. Трассировка в Langfuse и формирование сводки метрик
        telemetry_metrics = self.telemetry.log_rag_execution(
            query=clean_query,
            retrieved_docs=retrieved_docs,
            final_answer=generated_answer,
            search_latency_ms=search_latency_ms,
            llm_latency_ms=llm_latency_ms,
            usage_metadata=usage_metadata,
            system_prompt=self.llm_client.system_prompt,
            context_text=user_message,
            model_name=getattr(gen_result, "model_name", self.llm_client.model_name),
        )
        metrics_badge = self.telemetry.format_metrics_badge(telemetry_metrics)

        return {
            "query": clean_query,
            "answer": generated_answer,
            "retrieved_docs": retrieved_docs,
            "sources": sources,
            "metrics": telemetry_metrics,
            "metrics_badge": metrics_badge,
            "search_latency_sec": search_latency_sec,
            "llm_latency_sec": llm_latency_sec,
            "total_latency_sec": total_latency_sec,
        }


def run_demonstration() -> None:
    """
    Запуск проверочных тестов сквозного RAG:
    Тест 1 (Полноценный вопрос): 'Какова продолжительность ежегодного оплачиваемого отпуска?'
    Тест 2 (Проверка на галлюцинацию): 'Как зарегистрировать космический корабль в реестре Молдовы?'
    """
    service = RAGService()

    divider = "=" * 80
    sub_divider = "-" * 80

    test_queries = [
        (
            "Тест 1 (Полноценный вопрос по ТК РМ)",
            "Какова продолжительность ежегодного оплачиваемого отпуска?",
            "Ожидается извлечение Статьи 113 и точный ответ о 28 календарных днях с указанием источника.",
        ),
        (
            "Тест 2 (Проверка на галлюцинацию / Out-of-Domain)",
            "Как зарегистрировать космический корабль в реестре Молдовы?",
            "Ожидается строгий отказ: 'В предоставленном контексте информации недостаточно для ответа на данный вопрос.'",
        ),
    ]

    print("\n" + divider)
    print("ДЕМОНСТРАЦИЯ RAG: ГЕНЕРАЦИЯ ОТВЕТА GEMINI, ЗАЩИТА ОТ ГАЛЛЮЦИНАЦИЙ И ТЕЛЕМЕТРИЯ")
    print(divider)

    for idx, (test_name, query, expectation) in enumerate(test_queries, 1):
        print(f"\n{divider}")
        print(f"{test_name}")
        print(f"Ожидание: {expectation}")
        print(f"Вопрос: \"{query}\"")
        print(sub_divider)

        result = service.answer(query=query)

        print(f"Время выполнения: {result['total_latency_sec']:.2f} сек")
        print(f"Отобрано фрагментов в контекст: {len(result['retrieved_docs'])}")

        print("\n[СГЕНЕРИРОВАННЫЙ ОТВЕТ LLM]:")
        print(result["answer"])

        print("\n[ИЗВЛЕЧЕННЫЕ ИСТОЧНИКИ]:")
        if result["sources"]:
            for src in result["sources"]:
                print(f"  • {src}")
        else:
            print("  (Источники отсутствуют — в контексте не найдено подтверждающих норм)")

        if result.get("metrics_badge"):
            print("\n" + result["metrics_badge"])

        print(sub_divider)


def run_interactive(service: RAGService | None = None) -> None:
    """
    Интерактивный диалоговый режим (REPL):
    Позволяет пользователю вводить любые вопросы в реальном времени.
    Модели и индекс инициализируются один раз и остаются в оперативной памяти.
    """
    service = service or RAGService()
    divider = "=" * 80
    sub_divider = "-" * 80

    print("\n" + divider)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ RAG: ТРУДОВОЙ КОДЕКС РЕСПУБЛИКИ МОЛДОВА")
    print("Задайте любой вопрос по Трудовому кодексу.")
    print("Для завершения работы введите 'exit', 'quit' или 'q'.")
    print(divider + "\n")

    while True:
        try:
            query = input("\nВаш вопрос > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q", "выход"):
                print("\nСессия завершена.")
                break

            res = service.answer(query=query)

            print("\n" + sub_divider)
            print(f"[ОТВЕТ LLM] (время: {res['total_latency_sec']:.2f} сек, фрагментов в контексте: {len(res['retrieved_docs'])}):")
            print(sub_divider)
            print(res["answer"])

            print("\n[ИСТОЧНИКИ]:")
            if res["sources"]:
                for s in res["sources"]:
                    print(f"  • {s}")
            else:
                print("  (Источники не найдены или информации недостаточно)")

            if res.get("metrics_badge"):
                print("\n" + res["metrics_badge"])
            print(sub_divider)

        except (KeyboardInterrupt, EOFError):
            print("\nСессия завершена.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Сквозной запуск RAG-сервиса (Retriever -> FlashRank -> Gemini LLM)"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Задать конкретный пользовательский вопрос",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Запустить интерактивный диалоговый режим (REPL)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Принудительно запустить 2 эталонных проверочных теста",
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.query:
        service = RAGService()
        res = service.answer(args.query)
        print("\n" + "=" * 80)
        print(f"ВОПРОС: {res['query']}")
        print("=" * 80)
        print("\n[ОТВЕТ LLM]:")
        print(res["answer"])
        print("\n[ИСТОЧНИКИ]:")
        if res["sources"]:
            for s in res["sources"]:
                print(f"  • {s}")
        else:
            print("  (Источники не указаны или информации недостаточно)")
        if res.get("metrics_badge"):
            print("\n" + res["metrics_badge"])
        print("=" * 80 + "\n")
    else:
        run_demonstration()
        print("\nПодсказка:")
        print("  • Чтобы задать свой вопрос:        python -m src.generation.rag_service --query \"Ваш вопрос\"")
        print("  • Чтобы открыть интерактивный чат: python -m src.generation.rag_service -i\n")
