"""
storage.py — 持久化层
  - KnowledgeBase：线程安全的轻量知识库，基于关键词 Token 匹配做 RAG 检索
  - SessionStore ：线程安全的会话持久化，与业务逻辑解耦

变更说明：
  - now_iso() 改为公开函数，由外部模块直接导入，不再绕道 storage
  - KnowledgeDoc / SessionState 序列化改用各自的 to_dict / from_dict，
    消除 storage 层对数据结构内部细节的感知
  - _load / _save 提取为 _load_json / _save_json 辅助函数，逻辑更清晰
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from config import DATA_DIR, KB_FILE, SESSIONS_FILE
from models import KnowledgeDoc, SessionState
from logger import get_logger

logger = get_logger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 公共工具函数
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """返回当前时间的 ISO 8601 字符串（秒精度），供其他模块直接导入使用。"""
    return datetime.now().isoformat(timespec="seconds")


def tokenize(text: str) -> List[str]:
    """简单的中英文混合分词，用于 BM-like 关键词检索。"""
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())


def _load_json(path: str) -> Dict:
    """读取 JSON 文件，出错时返回空字典并记录日志。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("加载文件失败 %s: %s", path, exc)
        return {}


def _save_json(path: str, data: Any) -> None:
    """原子写入 JSON 文件（先写临时文件再 rename），出错时记录日志但不抛出。"""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.error("保存文件失败 %s: %s", path, exc)
        try:
            os.remove(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """线程安全的轻量知识库，基于关键词 Token 匹配做 RAG 检索。"""

    def __init__(self) -> None:
        self._docs: Dict[str, KnowledgeDoc] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(KB_FILE):
            return
        for k, v in _load_json(KB_FILE).items():
            self._docs[k] = KnowledgeDoc.from_dict(v)

    def _save(self) -> None:
        _save_json(KB_FILE, {k: v.to_dict() for k, v in self._docs.items()})

    def add_document(self, title: str, content: str) -> KnowledgeDoc:
        doc = KnowledgeDoc(
            id=uuid4().hex,
            title=title.strip(),
            content=content.strip(),
            tokens=set(tokenize(f"{title}\n{content}")),
            created_at=now_iso(),
        )
        with self._lock:
            self._docs[doc.id] = doc
            self._save()
        return doc

    def list_documents(self) -> List[KnowledgeDoc]:
        with self._lock:
            docs = list(self._docs.values())
        return sorted(docs, key=lambda d: d.created_at, reverse=True)

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            removed = self._docs.pop(doc_id, None) is not None
            if removed:
                self._save()
        return removed

    def search(self, query: str, top_k: int = 3) -> List[Tuple[KnowledgeDoc, int]]:
        """返回按相关性排序的 (doc, score) 列表。"""
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        with self._lock:
            docs = list(self._docs.values())
        scored = [(doc, len(query_tokens & doc.tokens)) for doc in docs]
        scored = [(doc, score) for doc, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """线程安全的会话持久化层，与业务逻辑解耦。"""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(SESSIONS_FILE):
            return
        for k, v in _load_json(SESSIONS_FILE).items():
            self._sessions[k] = SessionState.from_dict(v)

    def _save(self) -> None:
        _save_json(SESSIONS_FILE, {k: v.to_dict() for k, v in self._sessions.items()})

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(session_id)

    def list_all(self) -> List[SessionState]:
        with self._lock:
            sessions = list(self._sessions.values())
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def upsert(self, session: SessionState) -> None:
        with self._lock:
            self._sessions[session.id] = session
            self._save()

    def append_message(self, session: SessionState, message: Dict[str, Any]) -> None:
        """原子追加消息并持久化。"""
        with self._lock:
            session.messages.append(message)
            self._sessions[session.id] = session
            self._save()

    def pop_last_user_message(self, session: SessionState) -> None:
        """出现异常时回滚最后一条 user 消息，避免上下文污染。"""
        with self._lock:
            if session.messages and session.messages[-1]["role"] == "user":
                session.messages.pop()
                self._save()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            removed = self._sessions.pop(session_id, None) is not None
            if removed:
                self._save()
        return removed
