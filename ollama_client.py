"""
ollama_client.py — 统一入口（向后兼容）

其他模块只需 import 这一个文件，无需感知内部拆分结构。

    from ollama_client import ChatManager, KnowledgeBase, SessionState
"""

from config import (           # noqa: F401
    DATA_DIR,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    KB_FILE,
    LLM_MAX_RETRIES,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    SESSIONS_FILE,
    TAVILY_API_KEY,
    TOOL_DEFINITIONS,
    TOOL_TIMEOUT,
)
from models import KnowledgeDoc, SessionState          # noqa: F401
from storage import KnowledgeBase, SessionStore, now_iso  # noqa: F401
from tool.tools import ToolExecutor                         # noqa: F401
from tool.tool_parser import (                              # noqa: F401
    normalize_tool_call,
    extract_tool_calls_from_text,
)
from chat_manager import ChatManager                   # noqa: F401

