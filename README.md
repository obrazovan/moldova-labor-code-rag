# Moldova Labor Code RAG (RAG-система по Трудовому кодексу Молдовы)

Модульная Retrieval-Augmented Generation (RAG) система по законодательству Республики Молдова (Трудовой кодекс), реализованная на чистом Python без тяжелых фреймворков (LangChain / LlamaIndex).

## Стек технологий и архитектура

- **Язык**: Python 3.10+
- **Векторная БД**: ChromaDB (локальное хранилище)
- **Эмбеддинги**: Sentence-Transformers (легкие модели для CPU: `rubert-tiny2` / `bge`)
- **Reranker**: FlashRank / Cross-Encoder (локально на CPU)
- **LLM**: Google Gemini Flash через официальный API
- **Архитектура**: Чистый модульный пайплайн с разделением ответственности:
  - `src/grabber/` — Сбор и дедупликация нормативных документов
  - `src/preprocessing/` — Парсинг HTML, очистка структуры и чанкинг по статьям
  - `src/embeddings/` — Векторизация и управление векторным индексом
  - `src/retrieval/` — Семантический поиск и фильтрация
  - `src/reranking/` — Переранжирование кандидатов
  - `src/generation/` — Формирование контекста и вызов LLM с защитой от галлюцинаций
  - `src/evaluation/` — Набор тестов и метрики качества (Hit Rate, MRR)

---

## Структура репозитория

```text
├── configs/
│   └── sources.json             # Конфигурация источников документов
├── data/
│   ├── raw/                     # Исходные HTML-документы, метаданные и манифест
│   │   ├── codul_muncii.html    # Трудовой кодекс РМ (Закон № 154/2003)
│   │   ├── manifest.json        # Хеши и статус для дедупликации
│   │   └── metadata.json        # Метаданные загруженного акта
│   └── processed/               # Извлеченные и структурированные статьи (JSON)
├── experiments/                 # Скрипты сравнения гиперпараметров и графики
├── src/
│   ├── grabber/                 # Модуль сбора документов с legis.md
│   │   └── legis_grabber.py
│   ├── preprocessing/           # Модуль парсинга и чанкинга
│   ├── embeddings/              # Векторизация и работа с векторным индексом
│   ├── retrieval/               # Поиск документов
│   ├── reranking/               # Реранкинг результатов
│   ├── generation/              # Генерация ответа
│   └── evaluation/              # Оценка метрик
├── PROJECT_SPEC.md              # Архитектурные требования и спецификация
├── requirements.txt             # Зависимости проекта
└── README.md
```

---

## Этап 1: Сбор данных (Grabber)

В рамках первого этапа реализован отказоустойчивый модуль автоматического сбора данных с государственного портала [legis.md](https://www.legis.md/):
- **Управление языковой сессией**: Автоматическая инициализация сессии через `https://www.legis.md/languages/index/ru` с сохранением cookie (`ci_session`) для получения актуального текста кодекса на русском языке.
- **Сборка полного документа**: Корректное объединение базовой разметки и динамически подгружаемого тела акта (`showdetails/{doc_id}`).
- **Стабилизация хеша**: Очистка динамических токенов Cloudflare (`__CF$cv$params`), гарантирующая детерминированный SHA-256 хеш.
- **Дедупликация**: Проверка SHA-256 по `data/raw/manifest.json`. Если нормативный акт не изменялся, повторное скачивание пропускается.
- **Метаданные**: Фиксация даты загрузки, URL и контрольной суммы в `metadata.json`.

---

## Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/obrazovan/moldova-labor-code-rag.git
cd moldova-labor-code-rag
```

### 2. Создание виртуального окружения
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Запуск сбора данных (Grabber)
```bash
python -m src.grabber.legis_grabber
```
