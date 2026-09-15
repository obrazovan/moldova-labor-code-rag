"""Модуль для автоматического сбора нормативных правовых актов с портала legis.md.

Выполняет:
- Установку языковой сессии (https://www.legis.md/languages/index/ru) для получения актов на русском языке
- Загрузку HTML-страниц источников, заданных в configs/sources.json
- Обработку динамического контента (подтягивание тела документа showdetails/{doc_id})
- Дедупликацию документов на основе вычисления SHA-256 хеша
- Ведение манифеста (data/raw/manifest.json) и метаданных (data/raw/metadata.json)
"""

import hashlib
import http.cookiejar
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("legis_grabber")


class CustomRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Сохраняет пользовательские заголовки (User-Agent, Accept и т.д.) при редиректах.
    
    Стандартный HTTPRedirectHandler в Python удаляет заголовки при редиректе,
    что приводит к блокировке со стороны Cloudflare (403 Forbidden).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req:
            for k, v in req.headers.items():
                if k.lower() not in ("host", "content-length"):
                    new_req.add_header(k, v)
        return new_req


class LegisGrabber:
    """Граббер законодательных актов с портала legis.md."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        config_path: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        timeout: int = 30,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[2]

        self.config_path = config_path or (base_dir / "configs" / "sources.json")
        self.raw_dir = raw_dir or (base_dir / "data" / "raw")
        self.timeout = timeout

        self.manifest_path = self.raw_dir / "manifest.json"
        self.metadata_path = self.raw_dir / "metadata.json"

        # Создаем сессию с хранилищем cookie
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar),
            CustomRedirectHandler,
        )

        self._active_lang: Optional[str] = None

        # Гарантируем наличие директории data/raw
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def load_sources(self) -> List[Dict[str, Any]]:
        """Загрузка конфигурации источников."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "sources" in data:
            return data["sources"]
        else:
            return [data]

    def load_manifest(self) -> Dict[str, Any]:
        """Чтение текущего manifest.json."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8-sig") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Не удалось прочитать manifest.json: %s. Создается новый.", e)
        return {}

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Сохранение обновленного manifest.json."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def load_metadata(self) -> List[Dict[str, Any]]:
        """Чтение текущего metadata.json."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning("Не удалось прочитать metadata.json: %s. Создается новый.", e)
        return []

    def save_metadata(self, metadata: List[Dict[str, Any]]) -> None:
        """Сохранение обновленного metadata.json."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _http_get(self, url: str, max_retries: int = 3, retry_delay: float = 2.0) -> str:
        """Выполнение HTTP GET-запроса через настроенный opener с cookie и повторными попытками."""
        headers = {
            "User-Agent": self.DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,ro;q=0.8,en;q=0.7",
        }

        last_error: Optional[Exception] = None
        current_delay = retry_delay

        for attempt in range(1, max_retries + 1):
            req = urllib.request.Request(url, headers=headers)
            try:
                with self.opener.open(req, timeout=self.timeout) as resp:
                    status_code = resp.status
                    if status_code != 200:
                        raise urllib.error.HTTPError(
                            url, status_code, f"Неожиданный статус код: {status_code}", resp.headers, None
                        )
                    charset = resp.headers.get_content_charset() or "utf-8"
                    content_bytes = resp.read()
                    return content_bytes.decode(charset, errors="replace")
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Запрос к %s завершился ошибкой: %s. Повтор %d/%d через %.1f сек...",
                        url,
                        e,
                        attempt + 1,
                        max_retries,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    logger.error("Все попытки (%d) исчерпаны для %s. Ошибка: %s", max_retries, url, e)

        if last_error:
            raise last_error
        raise RuntimeError(f"Не удалось получить ответ от {url}")

    def set_language(self, lang: str = "ru") -> None:
        """Установка языка через предварительный запрос на https://www.legis.md/languages/index/{lang}.
        
        Это сохраняет сессионную куку ci_session, необходимую для получения
        русскоязычной версии акта через showdetails/{doc_id}.
        """
        if self._active_lang == lang:
            return

        lang_url = f"https://www.legis.md/languages/index/{lang}"
        logger.info("Установка языка '%s' через %s...", lang, lang_url)
        try:
            self._http_get(lang_url)
            self._active_lang = lang
            logger.info("Языковая кука успешно получена и сохранена в сессии.")
        except Exception as e:
            logger.warning("Не удалось переключить язык через %s: %s", lang_url, e)

    def fetch_document_html(
        self,
        url: str,
        doc_id: Optional[Any] = None,
        lang: Optional[str] = "ru",
    ) -> str:
        """Загрузка полного HTML-документа.
        
        1. Устанавливает куку языка в сессии.
        2. Скачивает базовую страницу.
        3. При наличии динамического контейнера подгружает showdetails/{doc_id} на нужном языке.
        """
        if lang:
            self.set_language(lang)
            time.sleep(1.0)

        html = self._http_get(url)

        # Проверяем, требует ли страница подгрузки детального текста
        details_match = re.search(r'showDetails\s*\(\s*null\s*,\s*[\'"]?(\d+)[\'"]?\s*\)', html)
        target_doc_id = doc_id or (details_match.group(1) if details_match else None)

        if target_doc_id and '<div id="details"></div>' in html:
            details_url = f"https://www.legis.md/cautare/showdetails/{target_doc_id}"
            logger.info("Подгрузка тела документа из %s", details_url)
            try:
                time.sleep(1.0)  # пауза между запросами к legis.md
                details_html = self._http_get(details_url)
                # Встраиваем тело документа в контейнер #details
                html = html.replace(
                    '<div id="details"></div>',
                    f'<div id="details">\n<!-- Embedded by LegisGrabber -->\n{details_html}\n</div>',
                )
            except Exception as e:
                logger.warning("Не удалось загрузить showdetails/%s: %s. Сохраняется базовая страница.", target_doc_id, e)

        # Удаляем динамические одноразовые скрипты Cloudflare для детерминированности хеша
        html = self.clean_dynamic_scripts(html)

        return html

    @staticmethod
    def clean_dynamic_scripts(html: str) -> str:
        """Удаляет динамические скрипты Cloudflare (__CF$cv$params), меняющиеся при каждом запросе."""
        pattern = r'<script>\(function\(\)\{function c\(\).*?<\/script>'
        return re.sub(pattern, '', html, flags=re.DOTALL)

    @staticmethod
    def calculate_sha256(content: str) -> str:
        """Вычисление SHA-256 хеша для переданной строки (UTF-8)."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def grab_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка одного источника: скачивание, дедупликация, сохранение."""
        source_id = str(source.get("id") or source.get("doc_id"))
        title = source.get("title", f"Document {source_id}")
        url = source.get("url", "")
        doc_id = source.get("doc_id")
        lang = source.get("lang", "ru")

        if not url:
            raise ValueError(f"Источник {source_id} не содержит обязательного поля 'url'")

        logger.info("Начало обработки [%s] '%s'...", source_id, title)
        logger.info("URL: %s (doc_id: %s, lang: %s)", url, doc_id, lang)

        # 1. Скачивание контента с учетом языковой куки
        content = self.fetch_document_html(url, doc_id=doc_id, lang=lang)
        current_sha256 = self.calculate_sha256(content)
        file_name = f"{source_id}.html"
        target_file = self.raw_dir / file_name

        manifest = self.load_manifest()
        prev_entry = manifest.get(source_id, {})
        prev_sha256 = prev_entry.get("sha256")

        # 2. Дедупликация
        if target_file.exists() and prev_sha256 == current_sha256:
            logger.info(
                "[%s] Документ уже загружен и не изменился (SHA-256: %s...). Загрузка пропущена.",
                source_id,
                current_sha256[:8],
            )
            downloaded_at = prev_entry.get("downloaded_at") or datetime.now(timezone.utc).isoformat()
        else:
            # Сохраняем файл
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(content)

            downloaded_at = datetime.now(timezone.utc).isoformat()
            logger.info(
                "[%s] Файл сохранен в %s (размер: %d байт, SHA-256: %s)",
                source_id,
                target_file.name,
                len(content.encode("utf-8")),
                current_sha256,
            )

            # Обновление manifest.json
            manifest[source_id] = {
                "id": source_id,
                "title": title,
                "file": file_name,
                "sha256": current_sha256,
                "downloaded_at": downloaded_at,
            }
            self.save_manifest(manifest)

        # 3. Обновление metadata.json
        metadata_list = self.load_metadata()
        meta_entry = {
            "id": source_id,
            "title": title,
            "url": url,
            "downloaded_at": downloaded_at,
            "sha256": current_sha256,
        }

        updated = False
        for i, item in enumerate(metadata_list):
            if str(item.get("id")) == source_id:
                metadata_list[i] = meta_entry
                updated = True
                break
        if not updated:
            metadata_list.append(meta_entry)

        self.save_metadata(metadata_list)
        return meta_entry

    def run(self) -> List[Dict[str, Any]]:
        """Запуск процесса сбора для всех источников из конфигурации."""
        sources = self.load_sources()
        logger.info("Загружено источников для обработки: %d", len(sources))

        results = []
        for src in sources:
            try:
                res = self.grab_source(src)
                results.append(res)
            except Exception as e:
                logger.error("Ошибка при обработке источника %s: %s", src.get("id"), e, exc_info=True)

        logger.info("Сбор данных завершен. Успешно обработано: %d/%d источников.", len(results), len(sources))
        return results


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

    grabber = LegisGrabber()
    grabber.run()
