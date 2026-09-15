"""Управляющий скрипт конвейера предобработки и чанкинга данных (Этап 2).

Цепочка обработки:
Raw HTML (data/raw/) -> Clean (cleaner.py) -> Parse (parser.py) -> Chunk (chunker.py) -> Processed JSON (data/processed/)

Генерируемые файлы:
- data/processed/chunks_by_article.json (основной рабочий набор)
- data/processed/chunks_fixed_size.json
- data/processed/chunks_fixed_overlap.json
- data/processed/chunks_by_paragraph.json

Вывод в консоль:
Сводная статистика (количество статей, количество чанков по стратегиям, средний размер чанка).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.preprocessing.cleaner import clean_html
    from src.preprocessing.parser import parse_articles
    from src.preprocessing.chunker import TextChunker
except ImportError:
    from cleaner import clean_html
    from parser import parse_articles
    from chunker import TextChunker

logger = logging.getLogger("preprocessing.pipeline")


class PreprocessingPipeline:
    """Конвейер предобработки: очистка HTML, парсинг статей и многостратегический чанкинг."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.raw_dir = raw_dir or (base_dir / "data" / "raw")
        self.processed_dir = processed_dir or (base_dir / "data" / "processed")
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.metadata_path = self.raw_dir / "metadata.json"
        self.manifest_path = self.raw_dir / "manifest.json"

        # Гарантируем наличие целевой директории
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = TextChunker(
            default_chunk_size=self.chunk_size,
            default_overlap=self.overlap,
        )

    def load_metadata(self, doc_id: str = "codul_muncii") -> Dict[str, Any]:
        """Загрузка метаданных документа из metadata.json или manifest.json."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if str(item.get("id")) == doc_id:
                                return item
                        if data:
                            return data[0]
                    elif isinstance(data, dict):
                        if doc_id in data:
                            return data[doc_id]
                        return data
            except Exception as e:
                logger.warning("Не удалось прочитать metadata.json: %s", e)

        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and doc_id in data:
                        return data[doc_id]
            except Exception as e:
                logger.warning("Не удалось прочитать manifest.json: %s", e)

        return {
            "id": doc_id,
            "title": "Трудовой кодекс Республики Молдова",
            "url": "https://www.legis.md/cautare/getResults?doc_id=155882&lang=ru",
        }

    def load_html(self, file_name: str = "codul_muncii.html") -> str:
        """Чтение исходного HTML-файла."""
        file_path = self.raw_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Исходный HTML-файл не найден: {file_path}")

        logger.info("Чтение сырого HTML из %s...", file_path.name)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _save_json(data: Any, target_path: Path) -> None:
        """Сохранение структуры данных в JSON с форматированием."""
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run(self, html_filename: str = "codul_muncii.html") -> Dict[str, Any]:
        """Запуск полного конвейера Clean -> Parse -> Chunk и сохранение результатов."""
        logger.info("=== Запуск Preprocessing Pipeline (Этап 2) ===")

        # 1. Загрузка исходных данных
        raw_html = self.load_html(html_filename)
        doc_id = Path(html_filename).stem
        metadata = self.load_metadata(doc_id=doc_id)

        # 2. Очистка HTML
        logger.info("1/3 Очистка HTML (Clean)...")
        clean_text = clean_html(raw_html, as_text=True)
        logger.info("Очистка завершена. Исходный размер: %d симв., очищенный текст: %d симв.", len(raw_html), len(clean_text))

        # 3. Парсинг структуры (Разделы, Главы, Статьи)
        logger.info("2/3 Синтаксический анализ структуры (Parse)...")
        articles = parse_articles(clean_text, metadata=metadata)
        logger.info("Парсинг завершен. Извлечено статей: %d", len(articles))

        # 4. Чанкинг по 4 стратегиям
        logger.info("3/3 Генерация чанков (Chunk) по 4 стратегиям...")

        strategies_config = [
            ("by_article", "chunks_by_article.json", {}),
            ("fixed_size", "chunks_fixed_size.json", {"chunk_size": self.chunk_size}),
            ("fixed_overlap", "chunks_fixed_overlap.json", {"chunk_size": self.chunk_size, "overlap": self.overlap}),
            ("by_paragraph", "chunks_by_paragraph.json", {}),
        ]

        stats_summary: Dict[str, Any] = {
            "document_id": doc_id,
            "document_title": metadata.get("title", ""),
            "articles_count": len(articles),
            "strategies": {},
        }

        for strategy_name, output_filename, kwargs in strategies_config:
            chunks = self.chunker.chunk_articles(articles, strategy=strategy_name, **kwargs)
            out_file = self.processed_dir / output_filename
            self._save_json(chunks, out_file)

            sizes = [len(c["text"]) for c in chunks] if chunks else [0]
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0
            min_size = min(sizes) if sizes else 0
            max_size = max(sizes) if sizes else 0

            stats_summary["strategies"][strategy_name] = {
                "file": output_filename,
                "file_path": str(out_file),
                "chunks_count": len(chunks),
                "avg_size_chars": round(avg_size, 1),
                "min_size_chars": min_size,
                "max_size_chars": max_size,
            }

            logger.info(
                "Стратегия '%s': сохранено %d чанков в %s (ср. размер: %.1f симв.)",
                strategy_name,
                len(chunks),
                output_filename,
                avg_size,
            )

        # 5. Вывод форматированной статистики в консоль
        self.print_summary(stats_summary)
        return stats_summary

    @staticmethod
    def print_summary(stats: Dict[str, Any]) -> None:
        """Вывод аккуратной и наглядной сводной таблицы результатов в консоль."""
        print("\n" + "=" * 78)
        print("  СВОДНАЯ СТАТИСТИКА ЭТАПА 2: PREPROCESSING & CHUNKING PIPELINE")
        print("=" * 78)
        print(f"  Документ:         {stats.get('document_title', 'Трудовой кодекс')}")
        print(f"  Идентификатор:    {stats.get('document_id', 'codul_muncii')}")
        print(f"  Найдено статей:   {stats.get('articles_count', 0)}")
        print("-" * 78)
        print(f"  {'Стратегия чанкинга':<18} | {'Чанков':<8} | {'Ср. размер':<12} | {'Мин..Макс':<10} | {'Файл'}")
        print("-" * 78)

        for strat_name, strat_data in stats.get("strategies", {}).items():
            count = strat_data.get("chunks_count", 0)
            avg_s = strat_data.get("avg_size_chars", 0.0)
            min_s = strat_data.get("min_size_chars", 0)
            max_s = strat_data.get("max_size_chars", 0)
            fname = strat_data.get("file", "")
            size_range = f"{min_s}..{max_s}"
            print(f"  {strat_name:<18} | {count:<8} | {avg_s:<12.1f} | {size_range:<10} | {fname}")

        print("-" * 78)
        print("  Все наборы чанков успешно сохранены в директорию: data/processed/")
        print("=" * 78 + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]

    pipeline = PreprocessingPipeline()
    pipeline.run()
