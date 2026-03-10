"""
storage.py — 持久化层
  - KnowledgeBase：线程安全的轻量知识库，基于关键词 Token 匹配做 RAG 检索
  - SessionStore ：线程安全的会话持久化，与业务逻辑解耦
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
# 工具函数（仅供本模块使用）
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _tokenize(text: str) -> List[str]:
    """简单的中英文混合分词，用于 BM-like 关键词检索。"""
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())


def _load_json_file(path: str) -> Dict:
    """读取 JSON 文件，出错时返回空字典并记录日志。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("加载文件失败 %s: %s", path, exc)
        return {}


def _save_json_file(path: str, data: Any) -> None:
    """写入 JSON 文件，出错时记录日志但不抛出。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("保存文件失败 %s: %s", path, exc)


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
        raw = _load_json_file(KB_FILE)
        for k, v in raw.items():
            v["tokens"] = set(v.get("tokens", []))
            self._docs[k] = KnowledgeDoc(**v)

    def _save(self) -> None:
        data = {k: {**v.__dict__, "tokens": list(v.tokens)} for k, v in self._docs.items()}
        _save_json_file(KB_FILE, data)

    def add_document(self, title: str, content: str) -> KnowledgeDoc:
        doc = KnowledgeDoc(
            id=uuid4().hex,
            title=title.strip(),
            content=content.strip(),
            tokens=set(_tokenize(f"{title}\n{content}")),
            created_at=_now_iso(),
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
        query_tokens = set(_tokenize(query))
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
        raw = _load_json_file(SESSIONS_FILE)
        for k, v in raw.items():
            self._sessions[k] = SessionState(**v)

    def _save(self) -> None:
        data = {k: v.__dict__ for k, v in self._sessions.items()}
        _save_json_file(SESSIONS_FILE, data)

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
