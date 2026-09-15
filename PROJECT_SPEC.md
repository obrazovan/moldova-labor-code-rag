# Проект: RAG-система по законодательству Республики Молдова (Трудовой кодекс)

## 1. Стек и ограничения
- Язык: Python 3.10+
- ОС: Windows 10/11
- Vector DB: ChromaDB (локальное хранилище)
- Embeddings: Sentence-Transformers (легкие модели для CPU: rubert-tiny2 или BGE)
- Reranker: FlashRank или Cross-Encoder (локально на CPU)
- LLM: Google Gemini Flash через официальный API (ключ из Google AI Studio)
- Никаких тяжелых абстракций вроде LangChain/LlamaIndex и никаких MCP. Весь pipeline пишется на чистом Python с модульной структурой.

## 2. Структура репозитория
rag_lab1/
 configs/           # Конфигурационные файлы (источники, параметры RAG)
 data/
    raw/           # Исходные HTML файлы и manifest.json от граббера
    processed/     # Очищенные JSON файлы со статьями и метаданными
 src/
    grabber/       # Автоматический сбор документов и дедупликация
    preprocessing/ # Парсинг HTML, очистка и чанкинг
    embeddings/    # Векторизация и работа с ChromaDB
    retrieval/     # Поиск документов и фильтрация (similarity threshold)
    reranking/     # Переранжирование результатов
    generation/    # Формирование промпта, вызов Gemini, анти-галлюцинации
    evaluation/    # Набор 30 тестов и расчет метрик (Hit Rate, MRR)
 experiments/       # Скрипты сравнения параметров и графики
 requirements.txt
 README.md

## 3. Источник данных
- Портал: legis.md
- Базовый документ: Трудовой кодекс Республики Молдова (ID: 141870, lang=ru).
- Формат: скачивание HTML, контроль по SHA-256 хешу, парсинг по статьям (Статья X = отдельный логический блок).
