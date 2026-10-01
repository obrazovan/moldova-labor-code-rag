"""
Модуль входного контроля запросов (Guardrails):
- Защита от Prompt Injection и Jailbreak атак (Security Guard)
- Гибридная проверка принадлежности вопроса к трудовому праву РМ (Domain Guard)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "configs"
    / "guardrails_cache.json"
)

# Обобщенный якорный текст предметной области трудового права
LABOR_DOMAIN_ANCHOR_TEXT = (
    "Трудовой кодекс Республики Молдова: трудовые отношения, трудовой договор, "
    "права и обязанности работника и работодателя, отпуск, увольнение, заработная плата, "
    "рабочее время, время отдыха, дисциплинарные взыскания, охрана труда."
)

# Контрастный якорный текст тем вне предметной области (Out-of-Domain)
OUT_OF_DOMAIN_ANCHOR_TEXT = (
    "кулинарные рецепты, приготовление блюд, супы, еда, астрономия, космос, галактики, "
    "уголовный кодекс, кража, убийство, налоговые ставки, программирование, погода, спорт."
)


class DomainGuard:
    """
    Класс входного Guardrail:
    1. Защита от попыток внедрения промптов (Prompt Injection / Jailbreak).
    2. Гибридная валидация предметной области (Трудовой кодекс Республики Молдова).
    """

    REJECTION_SECURITY = "Запрос отклонён политикой безопасности системы."
    REJECTION_DOMAIN = (
        "Данный ассистент отвечает исключительно на вопросы по Трудовому кодексу Республики Молдова. "
        "Пожалуйста, сформулируйте вопрос, связанный с трудовыми отношениями."
    )

    def __init__(
        self,
        config_path: str | Path | None = None,
        embedder: Any | None = None,
    ) -> None:
        """
        Инициализация Guardrail.

        :param config_path: Путь к файлу конфигурации configs/guardrails_cache.json.
        :param embedder: Экземпляр EmbeddingModel для вычисления семантических эмбеддингов.
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config(self.config_path)
        guard_cfg = self.config.get("guardrails", {})

        self.enabled = guard_cfg.get("enabled", True)
        self.injection_keywords: list[str] = [
            kw.lower() for kw in guard_cfg.get("injection_keywords", [])
        ]
        self.domain_keywords: list[str] = [
            kw.lower() for kw in guard_cfg.get("domain_keywords", [])
        ]
        self.semantic_domain_threshold: float = float(
            guard_cfg.get("semantic_domain_threshold", 0.40)
        )

        # Регулярные выражения для детекции Prompt Injection
        self.injection_patterns: list[re.Pattern] = [
            re.compile(r"ignore\s+(all\s+)?(previous\s+)?instructions?", re.IGNORECASE),
            re.compile(r"забудь\s+(все\s+)?(предыдущие\s+)?инструкции?", re.IGNORECASE),
            re.compile(r"system\s+prompt", re.IGNORECASE),
            re.compile(r"системный\s+промпт", re.IGNORECASE),
            re.compile(r"ты\s+теперь\s+", re.IGNORECASE),
            re.compile(r"act\s+as\s+", re.IGNORECASE),
            re.compile(r"\bjailbreak\b", re.IGNORECASE),
            re.compile(r"\boverride\b", re.IGNORECASE),
            re.compile(r"developer\s+mode", re.IGNORECASE),
            re.compile(r"режим\s+разработчика", re.IGNORECASE),
            re.compile(r"раскрой\s+промпт", re.IGNORECASE),
        ]

        # Явные маркеры нерелевантных доменов (кулинария, астрономия, УК и т.д.)
        self.out_of_domain_markers: list[str] = [
            "сварить", "приготовить", "рецепт", "зама", "заму", "борщ", "суп", "пирог",
            "ингредиент", "жарить", "выпекать", "кулинар",
            "луна", "солнце", "марса", "космос", "галактик", "орбит",
            "ст ук", "убийств", "краж", "грабеж", "ограблен", "ук рм",
        ]

        self.embedder = embedder
        self._labor_anchor_vec: np.ndarray | None = None
        self._out_anchor_vec: np.ndarray | None = None

        if self.embedder is not None:
            self._init_anchors()

    def _init_anchors(self) -> None:
        """Предварительный расчет векторов семантических якорей."""
        if self.embedder is not None and self._labor_anchor_vec is None:
            try:
                l_vec = self.embedder.encode_query(LABOR_DOMAIN_ANCHOR_TEXT)
                o_vec = self.embedder.encode_query(OUT_OF_DOMAIN_ANCHOR_TEXT)
                self._labor_anchor_vec = np.array(l_vec, dtype=np.float32)
                self._out_anchor_vec = np.array(o_vec, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Не удалось инициализировать векторные якоря DomainGuard: {e}")

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        """Загрузка JSON-конфигурации."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации {path}: {e}")
        return {}

    def check_security(self, query: str) -> tuple[bool, str]:
        """
        Проверка на попытки внедрения вредоносных инструкций (Prompt Injection).
        :param query: Входной запрос пользователя.
        :return: (True, "") если чисто, (False, REJECTION_SECURITY) если найдена атака.
        """
        lower_q = query.lower()

        # 1. Проверка по ключевым словам
        for kw in self.injection_keywords:
            if kw in lower_q:
                logger.warning(f"Guardrail Security: обнаружено ключевое слово injection '{kw}'")
                return False, self.REJECTION_SECURITY

        # 2. Проверка по регулярным выражениям
        for pattern in self.injection_patterns:
            if pattern.search(query):
                logger.warning(f"Guardrail Security: сработал injection pattern '{pattern.pattern}'")
                return False, self.REJECTION_SECURITY

        return True, ""

    def check_domain(
        self,
        query: str,
        query_embedding: list[float] | np.ndarray | None = None,
    ) -> tuple[bool, str]:
        """
        Гибридная проверка принадлежности к предметной области Трудового кодекса РМ.
        :param query: Текст запроса.
        :param query_embedding: Опционально предрассчитанный нормализованный эмбеддинг запроса.
        :return: (True, "") если относится к домену, (False, REJECTION_DOMAIN) если нет.
        """
        lower_q = query.lower()

        # 1. Быстрая лексическая проверка: поиск корней правовых терминов
        for kw in self.domain_keywords:
            if kw in lower_q:
                return True, ""

        # 2. Быстрая проверка явных маркеров чужих доменов (рецепты, космос и др.)
        for marker in self.out_of_domain_markers:
            if marker in lower_q:
                logger.info(f"Guardrail Domain: отклонен по маркеру нерелевантного домена '{marker}'")
                return False, self.REJECTION_DOMAIN

        # 3. Семантическая проверка по векторным якорям
        if query_embedding is None and self.embedder is not None:
            query_embedding = self.embedder.encode_query(query)

        if query_embedding is not None:
            if self._labor_anchor_vec is None:
                self._init_anchors()

            if self._labor_anchor_vec is not None and self._out_anchor_vec is not None:
                q_vec = np.array(query_embedding, dtype=np.float32)
                sim_labor = float(np.dot(q_vec, self._labor_anchor_vec))
                sim_out = float(np.dot(q_vec, self._out_anchor_vec))
                margin = sim_labor - sim_out

                logger.debug(
                    f"Guardrail Semantic: sim_labor={sim_labor:.4f}, sim_out={sim_out:.4f}, margin={margin:+.4f}"
                )

                # Если запрос сильнее похож на сторонние темы, чем на трудовое право
                if margin < 0.0 or sim_labor < self.semantic_domain_threshold:
                    logger.info(
                        f"Guardrail Domain: отклонен по семантике (margin={margin:+.4f}, sim_labor={sim_labor:.4f})"
                    )
                    return False, self.REJECTION_DOMAIN

        return True, ""

    def check(
        self,
        query: str,
        query_embedding: list[float] | np.ndarray | None = None,
    ) -> tuple[bool, str]:
        """
        Комплексная проверка запроса (Security + Domain).
        :param query: Текст запроса пользователя.
        :param query_embedding: Вектор эмбеддинга запроса (если уже вычислен).
        :return: (True, "") если запрос одобрен; (False, reason) если отклонен.
        """
        if not self.enabled:
            return True, ""

        clean_q = query.strip()
        if not clean_q:
            return False, "Запрос не может быть пустым."

        # 1. Проверка безопасности (Security Guard)
        sec_ok, sec_reason = self.check_security(clean_q)
        if not sec_ok:
            return False, sec_reason

        # 2. Проверка предметной области (Domain Guard)
        dom_ok, dom_reason = self.check_domain(clean_q, query_embedding=query_embedding)
        if not dom_ok:
            return False, dom_reason

        return True, ""
