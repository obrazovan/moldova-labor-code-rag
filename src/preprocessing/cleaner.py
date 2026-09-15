"""Модуль для очистки исходного HTML законодательных актов с портала legis.md.

Функции:
- clean_html(raw_html: str, as_text: bool = True) -> str:
  Извлекает контентный блок закона с помощью BeautifulSoup, удаляет служебные теги
  (<script>, <style>, кнопки, попапы, баннеры), нормализует пробелы и переводы строк.
"""

import logging
import re
from typing import Optional
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger("preprocessing.cleaner")


def clean_html(raw_html: str, as_text: bool = True) -> str:
    """Извлечение и очистка контентного блока закона из сырого HTML.

    Args:
        raw_html: Исходный HTML-код страницы законодательного акта.
        as_text: Если True, возвращает нормализованный структурированный текст.
                 Если False, возвращает очищенный фрагмент HTML-разметки.

    Returns:
        Очищенный текст документа (или очищенная HTML-разметка).
    """
    if not raw_html or not raw_html.strip():
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")

    # 1. Поиск основного контентного блока закона
    content_block: Optional[Tag] = (
        soup.find("div", id="contentdoc")
        or soup.find("div", class_="doc")
        or soup.find("div", class_="docx")
        or soup.find("div", id="details")
        or soup.find("body")
        or soup
    )

    if not content_block:
        logger.warning("Контентный блок документа не найден. Используется корневой элемент.")
        content_block = soup

    # 2. Удаление нежелательных тегов
    unwanted_tags = [
        "script",
        "style",
        "button",
        "form",
        "nav",
        "header",
        "footer",
        "noscript",
        "iframe",
        "svg",
        "input",
        "select",
    ]
    for tag in content_block.find_all(unwanted_tags):
        tag.decompose()

    # 3. Удаление служебных блоков по селекторам классов и ID (кнопки печати, попапы, модалки)
    service_pattern = re.compile(r"print|popup|btn|modal|share|banner|doclogo", re.IGNORECASE)
    for tag in content_block.find_all(attrs={"class": service_pattern}):
        tag.decompose()
    for tag in content_block.find_all(attrs={"id": service_pattern}):
        tag.decompose()

    # 4. Сохранение верхних индексов <sup> (например, Статья 7^1 или пункт b^1)
    # Это критично для предотвращения склеивания "Статья 7<sup>1</sup>" в "Статья 71"
    for sup in content_block.find_all("sup"):
        sup_content = sup.get_text().strip()
        if sup_content:
            sup.replace_with(f"^{sup_content}")
        else:
            sup.decompose()

    # 5. Замена тегов переноса строки <br> на символ \n
    for br in content_block.find_all(["br", "hr"]):
        br.replace_with("\n")

    if not as_text:
        # Возвращаем очищенный HTML-фрагмент
        cleaned_html_str = str(content_block)
        cleaned_html_str = cleaned_html_str.replace("\xa0", " ").replace("\u200b", "")
        cleaned_html_str = re.sub(r"[ \t]+", " ", cleaned_html_str)
        cleaned_html_str = re.sub(r"\n{3,}", "\n\n", cleaned_html_str)
        return cleaned_html_str.strip()

    # 6. Извлечение структурированного текста с сохранением блочных границ
    block_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "div", "blockquote"]
    blocks = []

    for elem in content_block.find_all(block_tags):
        # Проверяем, не является ли элемент контейнером других блочных элементов
        if elem.find(block_tags):
            continue

        text = elem.get_text(separator=" ")
        # Нормализация неразрывных пробелов и невидимых символов
        text = text.replace("\xa0", " ").replace("\u200b", "")
        # Схлопывание повторяющихся пробелов внутри строки
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            blocks.append(text)

    # Если через блочные элементы ничего не нашлось, извлекаем общий текст
    if not blocks:
        raw_text = content_block.get_text(separator="\n")
        raw_text = raw_text.replace("\xa0", " ").replace("\u200b", "")
        for line in raw_text.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                blocks.append(line)

    full_text = "\n".join(blocks)

    # 7. Финальная нормализация пробелов и переносов строк
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()


if __name__ == "__main__":
    from pathlib import Path
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sample_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "codul_muncii.html"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        cleaned = clean_html(raw)
        print(f"Очистка успешно завершена. Длина сырого HTML: {len(raw)}, длина текста: {len(cleaned)}")
        print("\n--- Первые 500 символов очищенного текста ---")
        print(cleaned[:500])
    else:
        print(f"Файл {sample_path} не найден.")
