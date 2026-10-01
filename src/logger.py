# src/logger.py — Professional Logging System
import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """إنشاء logger احترافي مع حفظ في ملف وعرض في Terminal"""
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger

    # تنسيق الرسائل
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # عرض في Terminal
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # حفظ في ملف
    today    = datetime.now().strftime("%Y-%m-%d")
    fh       = logging.FileHandler(log_dir / f"{today}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(fh)

    return logger