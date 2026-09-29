"""
Модуль взаимодействия с LLM Google Gemini через официальный Google GenAI SDK.
Обеспечивает вызовы моделей Gemini с контролем температуры, токенов и устойчивостью к сбоям сети/лимитов.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Настройка UTF-8 вывода для Windows консолей
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Подавляем предупреждение AFC SDK google.genai для чистоты логов
try:
    from google.genai.models import Models
    Models._logged_afc_warning = True
except Exception:
    pass

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "generation.json"


class GeminiClient:
    """
    Клиент для генерации текста через официальный SDK Google GenAI (Gemini).
    Поддерживает настройку системного промпта, температуры, лимита токенов,
    а также повторные попытки при временных ошибках сети и исчерпании лимитов.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """
        Инициализация клиента Gemini.

        :param config_path: Путь к файлу конфигурации configs/generation.json.
        :param api_key: Ключ API Gemini (если не передан, берется из переменной окружения GEMINI_API_KEY).
        :param model_name: Имя модели (если не передано, берется из GEMINI_MODEL или конфига).
        """
        # 1. Загрузка переменных окружения из .env
        load_dotenv()

        # 2. Проверка наличия API ключа
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "Переменная окружения GEMINI_API_KEY не найдена.\n"
                "Пожалуйста, создайте файл .env в корне проекта и добавьте строку:\n"
                "GEMINI_API_KEY=<ваш_ключ_google_ai_studio>\n"
                "Получить бесплатный API ключ можно на портале: https://aistudio.google.com/"
            )

        # 3. Загрузка параметров генерации из configs/generation.json
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config(self.config_path)

        # Приоритет model_name: аргумент -> GEMINI_MODEL в env -> config -> дефолт
        env_model = os.getenv("GEMINI_MODEL")
        self.model_name = (
            model_name
            or env_model
            or self.config.get("model_name", "gemini-2.5-flash")
        )

        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_output_tokens = int(self.config.get("max_output_tokens", 1024))
        self.system_prompt = self.config.get("system_prompt", "")

        # 4. Инициализация официального клиента Google GenAI
        try:
            from google import genai
            self._genai = genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info(
                f"GeminiClient успешно инициализирован: модель='{self.model_name}', "
                f"temperature={self.temperature}, max_tokens={self.max_output_tokens}"
            )
        except Exception as e:
            logger.error(f"Не удалось инициализировать клиент google.genai: {e}", exc_info=True)
            raise RuntimeError(f"Ошибка инициализации Google GenAI Client: {e}") from e

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        """Загрузка конфигурационного JSON-файла."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось прочитать файл конфигурации {path}: {e}")
        else:
            logger.warning(f"Файл конфигурации не найден: {path}. Будут использованы значения по умолчанию.")
        return {}

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        max_retries: int = 4,
        initial_delay: float = 2.0,
    ) -> str:
        """
        Отправка запроса к модели Gemini с переданным промптом и системной инструкцией.

        :param prompt: Пользовательский запрос (включая контекст статей).
        :param system_instruction: Системная инструкция (если None, берется system_prompt из конфига).
        :param max_retries: Максимальное количество попыток при сетевых ошибках и лимитах (503/429).
        :param initial_delay: Начальная задержка экспоненциального бэкоффа в секундах.
        :return: Сгенерированный текстовый ответ.
        """
        if not prompt or not prompt.strip():
            logger.warning("Передан пустой промпт в метод generate(). Возврат пустой строки.")
            return ""

        from google.genai import types

        sys_inst = system_instruction if system_instruction is not None else self.system_prompt
        gen_config = types.GenerateContentConfig(
            system_instruction=sys_inst if sys_inst else None,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        current_model = self.model_name
        delay = initial_delay

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Отправка запроса в Gemini [{current_model}] (попытка {attempt}/{max_retries})..."
                )
                response = self.client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=gen_config,
                )

                # Извлечение сгенерированного текста
                if response.text is not None:
                    return response.text.strip()

                # Если response.text равен None (например, при специфическом finish_reason)
                if response.candidates and response.candidates[0].content:
                    parts = response.candidates[0].content.parts or []
                    extracted_text = "".join(
                        getattr(part, "text", "") for part in parts if hasattr(part, "text")
                    )
                    return extracted_text.strip()

                return ""

            except Exception as exc:
                err_str = str(exc)
                logger.warning(
                    f"Ошибка при вызове Gemini API (попытка {attempt}/{max_retries}): {exc}"
                )

                # Обработка 404: если модель устарела или недоступна для данного ключа, пробуем fallback
                if "404" in err_str and ("no longer available" in err_str or "NOT_FOUND" in err_str):
                    fallback_model = "gemini-3.5-flash" if current_model != "gemini-3.5-flash" else "gemini-3.8-flash"
                    logger.warning(
                        f"Модель '{current_model}' недоступна (404 Not Found). "
                        f"Автоматическое переключение на fallback модель '{fallback_model}'..."
                    )
                    current_model = fallback_model
                    self.model_name = fallback_model
                    continue

                # Обработка 503 High Demand: переключение на стабильную модель линейки Flash
                if "503" in err_str and "high demand" in err_str.lower():
                    backup_model = "gemini-3.5-flash" if current_model != "gemini-3.5-flash" else "gemini-3.8-flash"
                    logger.warning(
                        f"Модель '{current_model}' перегружена (503 High Demand). "
                        f"Переключение на резервную модель '{backup_model}'..."
                    )
                    current_model = backup_model
                    time.sleep(1.0)
                    continue

                # Если это последняя попытка — логируем ошибку и пробрасываем исключение
                if attempt == max_retries:
                    logger.error(
                        f"Исчерпано количество попыток ({max_retries}) обращения к Gemini API: {exc}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Не удалось получить ответ от Gemini API после {max_retries} попыток: {exc}"
                    ) from exc

                # Экспоненциальный бэкофф при других временных ошибках
                logger.info(f"Ожидание {delay:.1f} сек перед повторной попыткой...")
                time.sleep(delay)
                delay *= 2.0

        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Проверка инициализации GeminiClient...")
    try:
        client = GeminiClient()
        test_answer = client.generate("Ответь строго одним словом: 'РАБОТАЕТ'")
        print(f"Ответ модели: {test_answer}")
    except Exception as e:
        print(f"Ошибка: {e}")
