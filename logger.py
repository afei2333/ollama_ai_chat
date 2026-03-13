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
import shutil
import sys
from logging.handlers import TimedRotatingFileHandler


class _WinSafeRotatingHandler(TimedRotatingFileHandler):
    """
    Windows 兼容的按天滚动日志 Handler。

    Windows 根本问题：os.rename() 要求文件没有任何打开的句柄，
    但 FastAPI 的线程池中多个线程可能同时持有文件句柄在写日志。
    即使在 doRollover 里先 close() 自己的 stream，其他线程的 stream
    句柄（指向同一底层文件）仍然存在，rename 依然失败。

    解决方案：重写 rotate()，用 copy+truncate 代替 rename：
      1. shutil.copy2(source, dest)  — 把当前内容复制到归档文件
      2. open(source, 'w').truncate()— 清空原文件（不重命名，句柄继续有效）
    其他线程的句柄继续指向原文件路径，写入的新内容直接进入下一天的日志，
    行为与 rename 方案完全一致，且对 Windows 文件锁友好。

    非 Windows 平台走标准 rename 逻辑，无任何影响。
    """

    def rotate(self, source: str, dest: str) -> None:
        if sys.platform != "win32":
            super().rotate(source, dest)
            return

        # copy 当前日志内容到归档目标
        shutil.copy2(source, dest)
        # 清空原文件（保留路径和所有打开的句柄）
        with open(source, "w", encoding="utf-8"):
            pass


LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ── 格式 ────────────────────────────────────────────────────────────────────
_FMT_VERBOSE = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_FMT_CONSOLE = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"


def _file_handler(filename: str, level: int = logging.DEBUG) -> _WinSafeRotatingHandler:
    """按天滚动，保留 30 天，UTF-8 编码，Windows 文件锁安全。"""
    handler = _WinSafeRotatingHandler(
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
        return
    root.setLevel(logging.DEBUG)
    root.addHandler(_console_handler(logging.INFO))
    root.addHandler(_file_handler("app.log", logging.DEBUG))


def _setup_llm_logger() -> None:
    """LLM 专用 logger，不向上传播到 root（避免重复写入 app.log）。"""
    llm = logging.getLogger("llm")
    if llm.handlers:
        return
    llm.setLevel(logging.DEBUG)
    llm.propagate = False
    llm.addHandler(_file_handler("llm.log", logging.DEBUG))
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