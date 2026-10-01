"""
Корневой файл запуска интерактивного Web UI на Streamlit:
streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# Подавление фоновых предупреждений
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Обеспечиваем доступ ко всем пакетам проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui.app import main

if __name__ == "__main__":
    main()
