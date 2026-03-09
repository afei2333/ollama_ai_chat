"""
logger.py — 统一日志配置模块

日志策略：
  - 控制台：INFO 级别，简洁格式，方便开发时实时查看
  - 文件：DEBUG 级别，按天滚动，保留 30 天，记录完整上下文
  - 大模型原始请求/响应单独写入 llm.log，避免与业务日志混杂

目录结构：
  logs/
  ├── app.log        # 业务日志（按天滚动）
  └── llm.log        # LLM 原始输入输出（按天滚动）

使用方式：
  from logger import get_logger, get_llm_logger
  log     = get_logger(__name__)
  llm_log = get_llm_logger()
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ── 格式 ────────────────────────────────────────────────────────────────────
_FMT_VERBOSE = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_FMT_CONSOLE = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"


def _file_handler(filename: str, level: int = logging.DEBUG) -> TimedRotatingFileHandler:
    """按天滚动，保留 30 天，UTF-8 编码。"""
    handler = TimedRotatingFileHandler(
        filename=os.path.join(LOGS_DIR, filename),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FMT_VERBOSE, datefmt=_DATE_FMT))
    return handler


def _console_handler(level: int = logging.INFO) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FMT_CONSOLE, datefmt=_DATE_FMT))
    return handler


def _setup_root_logger() -> None:
    """只在首次调用时初始化根 logger，避免重复添加 handler。"""
    root = logging.getLogger()
    if root.handlers:
        return  # 已初始化，跳过
    root.setLevel(logging.DEBUG)
    root.addHandler(_console_handler(logging.INFO))
    root.addHandler(_file_handler("app.log", logging.DEBUG))


def _setup_llm_logger() -> None:
    """LLM 专用 logger，不向上传播到 root（避免重复写入 app.log）。"""
    llm = logging.getLogger("llm")
    if llm.handlers:
        return
    llm.setLevel(logging.DEBUG)
    llm.propagate = False                          # 不传播到 root
    llm.addHandler(_file_handler("llm.log", logging.DEBUG))
    # LLM 日志也输出到控制台，但只显示 WARNING 以上，避免刷屏
    llm.addHandler(_console_handler(logging.WARNING))


# 模块加载时立即初始化
_setup_root_logger()
_setup_llm_logger()


def get_logger(name: str) -> logging.Logger:
    """获取业务 logger，name 通常传 __name__。"""
    return logging.getLogger(name)


def get_llm_logger() -> logging.Logger:
    """获取专用于记录 LLM 原始请求/响应的 logger。"""
    return logging.getLogger("llm")