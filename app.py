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

# Заглушка для torchvision, чтобы Streamlit watcher не падал при инспекции transformers
if "torchvision" not in sys.modules:
    try:
        import torchvision
    except ImportError:
        tv = types.ModuleType("torchvision")
        tv.transforms = types.ModuleType("torchvision.transforms")
        tv.transforms.v2 = types.ModuleType("torchvision.transforms.v2")
        tv.transforms.v2.functional = types.ModuleType("torchvision.transforms.v2.functional")
        tv.io = types.ModuleType("torchvision.io")
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = tv.transforms
        sys.modules["torchvision.transforms.v2"] = tv.transforms.v2
        sys.modules["torchvision.transforms.v2.functional"] = tv.transforms.v2.functional
        sys.modules["torchvision.io"] = tv.io

# Обеспечиваем доступ ко всем пакетам проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui.app import main

if __name__ == "__main__":
    main()
