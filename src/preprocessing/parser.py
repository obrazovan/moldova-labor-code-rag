"""Модуль синтаксического анализа структуры закона (разделы, главы, статьи).

Функции:
- parse_articles(clean_html: str, metadata: dict) -> list[dict]:
  Извлекает с помощью регулярных выражений разделы, главы и все статьи закона,
  сопоставляя каждой статье номер, заголовок, текст, раздел, главу и метаданные.
"""

import logging
import re
from typing import Any, Dict, List, Optional

try:
    from src.preprocessing.cleaner import clean_html
except ImportError:
    from cleaner import clean_html

logger = logging.getLogger("preprocessing.parser")

# Регулярные выражения для поиска структурных элементов закона
RE_SECTION = re.compile(r"^РАЗДЕЛ\s+([IVXLCDM\d]+)", re.IGNORECASE)
RE_CHAPTER = re.compile(r"^Глава\s+([IVXLCDM\d]+(?:\s*\d+)?)", re.IGNORECASE)
RE_ARTICLE = re.compile(r"^Статья\s+(\d+(?:\s*\^\s*\d+)?)\.?(.*)", re.IGNORECASE)
RE_PART_ITEM = re.compile(r"^\(\d+", re.IGNORECASE)


def parse_articles(text_or_html: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Парсинг статей, разделов и глав законодательного акта.

    Args:
        text_or_html: Очищенный текст закона или HTML-разметка.
        metadata: Словарь метаданных документа (id, url, title, downloaded_at и др.).

    Returns:
        Список словарей, каждый из которых представляет распарсенную статью закона:
        - number: номер статьи (например, "1", "7^1", "392")
        - title: заголовок статьи (например, "Основные понятия")
        - text: полный текст статьи (заголовок + тело)
        - body: текст статьи без заголовка
        - paragraphs: список абзацев/пунктов статьи
        - section: текущий раздел (например, "РАЗДЕЛ I. ОБЩИЕ ПОЛОЖЕНИЯ")
        - chapter: текущая глава (например, "Глава I. ВВОДНЫЕ ПОЛОЖЕНИЯ")
        - document_id, source, url, downloaded_at, document_title
    """
    if metadata is None:
        metadata = {}

    # Если передан HTML с тегами, предварительно прогоняем через clean_html
    if "<html" in text_or_html.lower() or "<div" in text_or_html.lower() or "<p" in text_or_html.lower():
        clean_text = clean_html(text_or_html, as_text=True)
    else:
        clean_text = text_or_html

    # Разбиваем текст на строки / абзацы
    raw_lines = [line.strip() for line in clean_text.splitlines()]
    lines = [line for line in raw_lines if line]

    doc_id = str(metadata.get("id") or metadata.get("doc_id") or "codul_muncii")
    source = str(metadata.get("source") or metadata.get("id") or "codul_muncii")
    url = str(metadata.get("url") or "")
    downloaded_at = str(metadata.get("downloaded_at") or metadata.get("updated_at") or "")
    document_title = str(metadata.get("title") or "Трудовой кодекс Республики Молдова")

    current_section = ""
    current_chapter = ""
    articles: List[Dict[str, Any]] = []

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # 1. Проверка на начало Раздела
        m_sec = RE_SECTION.match(line)
        if m_sec:
            sec_num = m_sec.group(1)
            sec_rest = line[m_sec.end():].strip().lstrip(". -—:").strip()
            sec_title = sec_rest

            # Сбрасываем текущую главу при начале нового раздела
            current_chapter = ""

            # Проверяем, не вынесено ли название раздела на следующую(ие) строку(и)
            if i + 1 < n and not RE_SECTION.match(lines[i + 1]) and not RE_CHAPTER.match(lines[i + 1]) and not RE_ARTICLE.match(lines[i + 1]):
                next_l = lines[i + 1]
                if next_l.isupper() or len(next_l) < 120:
                    sec_title = (sec_title + " " + next_l).strip() if sec_title else next_l
                    i += 1
                    # Вторая строка названия раздела (например, 'В СФЕРЕ ТРУДА')
                    if i + 1 < n and not RE_SECTION.match(lines[i + 1]) and not RE_CHAPTER.match(lines[i + 1]) and not RE_ARTICLE.match(lines[i + 1]):
                        next_l2 = lines[i + 1]
                        if next_l2.isupper():
                            sec_title = sec_title + " " + next_l2
                            i += 1

            current_section = f"РАЗДЕЛ {sec_num}" + (f". {sec_title}" if sec_title else "")
            i += 1
            continue

        # 2. Проверка на начало Главы
        m_chap = RE_CHAPTER.match(line)
        if m_chap:
            chap_num = m_chap.group(1).replace(" ", "")
            chap_rest = line[m_chap.end():].strip().lstrip(". -—:").strip()
            chap_title = chap_rest

            if i + 1 < n and not RE_SECTION.match(lines[i + 1]) and not RE_CHAPTER.match(lines[i + 1]) and not RE_ARTICLE.match(lines[i + 1]):
                next_l = lines[i + 1]
                if next_l.isupper() or (len(next_l) < 120 and not next_l.startswith("(")):
                    chap_title = (chap_title + " " + next_l).strip() if chap_title else next_l
                    i += 1

            current_chapter = f"Глава {chap_num}" + (f". {chap_title}" if chap_title else "")
            i += 1
            continue

        # 3. Проверка на начало Статьи
        m_art = RE_ARTICLE.match(line)
        if m_art:
            art_num = m_art.group(1).replace(" ", "")
            art_title = m_art.group(2).strip().lstrip(". -—:").strip()
            i += 1

            # Склеиваем многострочные названия статьи (переносы из Word/HTML)
            while i < n:
                next_l = lines[i]
                if RE_SECTION.match(next_l) or RE_CHAPTER.match(next_l) or RE_ARTICLE.match(next_l):
                    break

                is_continuation = False
                # Если строка начинается с маленькой буквы — это продолжение названия
                if next_l and next_l[0].islower():
                    is_continuation = True
                # Если в скобках уточнение (не часть закона вида '(1)')
                elif next_l.startswith("(") and not RE_PART_ITEM.match(next_l):
                    is_continuation = True
                # Если заголовок пока пуст, а строка короткая и не начинается с номера части
                elif not art_title and not next_l.startswith("(") and len(next_l) < 120 and not next_l.endswith("."):
                    is_continuation = True

                if is_continuation:
                    art_title = (art_title + " " + next_l).strip()
                    i += 1
                else:
                    break

            # Сбор текста и абзацев статьи
            body_paragraphs: List[str] = []
            while i < n:
                next_l = lines[i]
                if RE_SECTION.match(next_l) or RE_CHAPTER.match(next_l) or RE_ARTICLE.match(next_l):
                    break
                body_paragraphs.append(next_l)
                i += 1

            # Формирование полного текста статьи
            full_header = f"Статья {art_num}."
            if art_title:
                full_header += f" {art_title}"

            if body_paragraphs:
                full_text = f"{full_header}\n\n" + "\n\n".join(body_paragraphs)
            else:
                full_text = full_header

            article_dict: Dict[str, Any] = {
                "number": art_num,
                "article_number": art_num,
                "title": art_title,
                "text": full_text,
                "body": "\n\n".join(body_paragraphs),
                "paragraphs": body_paragraphs,
                "section": current_section,
                "chapter": current_chapter,
                "document_id": doc_id,
                "source": source,
                "url": url,
                "downloaded_at": downloaded_at,
                "document_title": document_title,
            }
            articles.append(article_dict)
            continue

        i += 1

    logger.info("Парсинг завершен. Извлечено %d статей.", len(articles))
    return articles


if __name__ == "__main__":
    from pathlib import Path
    import json
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base_dir = Path(__file__).resolve().parents[2]
    raw_html_path = base_dir / "data" / "raw" / "codul_muncii.html"
    meta_path = base_dir / "data" / "raw" / "metadata.json"

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            metadata = data[0] if isinstance(data, list) and data else data

    if raw_html_path.exists():
        with open(raw_html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        parsed = parse_articles(html, metadata)
        print(f"Успешно распарсено {len(parsed)} статей.")
        print("\n--- Пример первой статьи ---")
        first = parsed[0]
        print(f"Статья {first['number']}: {first['title']}")
        print(f"Раздел: {first['section']}")
        print(f"Глава: {first['chapter']}")
        print(f"Текст (первые 200 симв.):\n{first['text'][:200]}...")
    else:
        print(f"Файл {raw_html_path} не найден.")
