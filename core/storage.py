"""
storage.py — 持久化层（SQLite 版）

变更说明：
  - KnowledgeBase 和 SessionStore 的存储后端由 JSON 文件改为 SQLite
  - 对外接口（方法签名、返回类型）与原 JSON 版完全一致，上层代码零改动
  - SQLite 使用 WAL 模式提升并发读性能；写操作加 threading.Lock 保持线程安全
  - 消息列表（messages）和知识库 tokens 仍序列化为 JSON 字符串存入 TEXT 列
  - now_iso / tokenize 等公共工具函数保持不变
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from config import DATA_DIR
from .models import KnowledgeDoc, SessionState
from .logger import get_logger

logger = get_logger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "assistant.db")


# ---------------------------------------------------------------------------
# 公共工具函数（接口不变）
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """返回当前时间的 ISO 8601 字符串（秒精度），供其他模块直接导入使用。"""
    return datetime.now().isoformat(timespec="seconds")


def tokenize(text: str) -> List[str]:
    """简单的中英文混合分词，用于 BM-like 关键词检索。"""
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())


# ---------------------------------------------------------------------------
# 数据库初始化
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """
    创建一个 SQLite 连接并开启 WAL 模式。
    每次调用返回新连接（适合多线程：每线程独立连接）。
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # 支持按列名访问
    conn.execute("PRAGMA journal_mode=WAL") # 提升并发读性能
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    """建表（如果不存在）。只在模块首次加载时调用一次。"""
    conn = _get_conn()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS kb_docs (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                content    TEXT NOT NULL,
                tokens     TEXT NOT NULL,   -- JSON 字符串，存 list[str]
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                model         TEXT NOT NULL,
                max_turns     INTEGER NOT NULL DEFAULT 8,
                system_prompt TEXT NOT NULL DEFAULT '',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );

            -- 消息单独拆表，便于后续按 session 查询、分页、搜索
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT,
                extra      TEXT,           -- JSON，存 tool_calls / name / _google_raw_content 等额外字段
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);
        """)
    conn.close()


_init_db()  # 模块加载时建表


# ---------------------------------------------------------------------------
# 内部辅助：消息行 ↔ dict 互转
# ---------------------------------------------------------------------------

def _msg_to_row(msg: Dict[str, Any], session_id: str) -> Tuple:
    """将业务层消息 dict 拆分为数据库列。

    注意：_google_raw_content 在 chat_manager 里已经 serialize_google_content 成 dict，
    但保险起见对每个 extra 字段单独做 JSON 可序列化检测，跳过无法序列化的值，
    防止 SDK 对象残留导致事务回滚、消息丢失。
    """
    role    = msg.get("role", "")
    content = msg.get("content") or ""

    # 逐字段检测，跳过不可序列化的值（如 Gemini SDK Content 对象）
    extra_keys: Dict[str, Any] = {}
    for k, v in msg.items():
        if k in ("role", "content"):
            continue
        try:
            json.dumps(v, ensure_ascii=False)
            extra_keys[k] = v
        except (TypeError, ValueError):
            logger.warning("消息字段 '%s' 无法 JSON 序列化，已跳过存储", k)

    extra = json.dumps(extra_keys, ensure_ascii=False) if extra_keys else None
    return (session_id, role, content, extra, now_iso())


def _row_to_msg(row: sqlite3.Row) -> Dict[str, Any]:
    """将数据库行还原为业务层消息 dict。"""
    msg: Dict[str, Any] = {
        "role":    row["role"],
        "content": row["content"] or "",
    }
    if row["extra"]:
        try:
            extra = json.loads(row["extra"])
            msg.update(extra)
        except json.JSONDecodeError:
            pass
    return msg


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """线程安全的轻量知识库，存储后端改为 SQLite。对外接口与原 JSON 版完全一致。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # 内存缓存：避免每次检索都查 DB（知识库一般数据量小）
        self._cache: Dict[str, KnowledgeDoc] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT id, title, content, tokens, created_at FROM kb_docs"
            ).fetchall()
            for row in rows:
                tokens = set(json.loads(row["tokens"]))
                self._cache[row["id"]] = KnowledgeDoc(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    tokens=tokens,
                    created_at=row["created_at"],
                )
        except Exception as exc:
            logger.warning("知识库缓存加载失败: %s", exc)
        finally:
            conn.close()

    def add_document(self, title: str, content: str) -> KnowledgeDoc:
        doc = KnowledgeDoc(
            id=uuid4().hex,
            title=title.strip(),
            content=content.strip(),
            tokens=set(tokenize(f"{title}\n{content}")),
            created_at=now_iso(),
        )
        tokens_json = json.dumps(list(doc.tokens), ensure_ascii=False)

        with self._lock:
            conn = _get_conn()
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO kb_docs (id, title, content, tokens, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (doc.id, doc.title, doc.content, tokens_json, doc.created_at),
                    )
                self._cache[doc.id] = doc
            finally:
                conn.close()
        return doc

    def list_documents(self) -> List[KnowledgeDoc]:
        with self._lock:
            docs = list(self._cache.values())
        return sorted(docs, key=lambda d: d.created_at, reverse=True)

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._cache:
                return False
            conn = _get_conn()
            try:
                with conn:
                    conn.execute("DELETE FROM kb_docs WHERE id = ?", (doc_id,))
                del self._cache[doc_id]
            finally:
                conn.close()
        return True

    def search(self, query: str, top_k: int = 3) -> List[Tuple[KnowledgeDoc, int]]:
        """返回按相关性排序的 (doc, score) 列表。"""
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        with self._lock:
            docs = list(self._cache.values())
        scored = [(doc, len(query_tokens & doc.tokens)) for doc in docs]
        scored = [(doc, score) for doc, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """
    线程安全的会话持久化层，存储后端改为 SQLite。
    对外接口与原 JSON 版完全一致，上层代码零改动。

    设计说明：
      - sessions 表存会话元数据
      - messages 表存消息（含外键级联删除）
      - 内存缓存 _cache 存 SessionState 对象（含已加载的 messages 列表）
      - 写操作加锁 + 事务，保证线程安全和原子性
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, SessionState] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """启动时将所有会话（含消息）读入内存缓存。"""
        conn = _get_conn()
        try:
            session_rows = conn.execute(
                "SELECT id, name, model, max_turns, system_prompt, created_at, updated_at "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()

            for srow in session_rows:
                sid = srow["id"]
                msg_rows = conn.execute(
                    "SELECT role, content, extra FROM messages "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (sid,),
                ).fetchall()
                messages = [_row_to_msg(r) for r in msg_rows]

                self._cache[sid] = SessionState(
                    id=sid,
                    name=srow["name"],
                    model=srow["model"],
                    max_turns=srow["max_turns"],
                    system_prompt=srow["system_prompt"],
                    created_at=srow["created_at"],
                    updated_at=srow["updated_at"],
                    messages=messages,
                )
        except Exception as exc:
            logger.warning("会话缓存加载失败: %s", exc)
        finally:
            conn.close()

    # ── 对外接口（与原 JSON 版签名完全一致） ──────────────────────────────

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._cache.get(session_id)

    def list_all(self) -> List[SessionState]:
        with self._lock:
            sessions = list(self._cache.values())
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def upsert(self, session: SessionState) -> None:
        """新建或更新会话元数据（不含消息）。"""
        with self._lock:
            conn = _get_conn()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO sessions
                            (id, name, model, max_turns, system_prompt, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            name          = excluded.name,
                            model         = excluded.model,
                            max_turns     = excluded.max_turns,
                            system_prompt = excluded.system_prompt,
                            updated_at    = excluded.updated_at
                        """,
                        (
                            session.id, session.name, session.model,
                            session.max_turns, session.system_prompt,
                            session.created_at, session.updated_at,
                        ),
                    )
                # ★ 关键修复：如果缓存里已有该 session（含历史消息），
                #   只更新元数据字段，不能用传入的空 messages 覆盖已有消息列表
                if session.id in self._cache:
                    cached = self._cache[session.id]
                    cached.name          = session.name
                    cached.model         = session.model
                    cached.max_turns     = session.max_turns
                    cached.system_prompt = session.system_prompt
                    cached.updated_at    = session.updated_at
                else:
                    self._cache[session.id] = session
            finally:
                conn.close()

    def append_message(self, session: SessionState, message: Dict[str, Any]) -> None:
        """原子追加消息并持久化（同时更新 session.updated_at）。"""
        row = _msg_to_row(message, session.id)
        with self._lock:
            conn = _get_conn()
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO messages (session_id, role, content, extra, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        row,
                    )
                    # 同步更新 session 的 updated_at
                    ts = now_iso()
                    conn.execute(
                        "UPDATE sessions SET updated_at = ? WHERE id = ?",
                        (ts, session.id),
                    )
                session.messages.append(message)
                session.updated_at = ts
                self._cache[session.id] = session
            finally:
                conn.close()

    def pop_last_user_message(self, session: SessionState) -> None:
        """出现异常时回滚最后一条 user 消息，避免上下文污染。"""
        with self._lock:
            if not session.messages or session.messages[-1].get("role") != "user":
                return
            conn = _get_conn()
            try:
                with conn:
                    # 删除该 session 最后一条 role='user' 的消息行
                    conn.execute(
                        """
                        DELETE FROM messages WHERE id = (
                            SELECT id FROM messages
                            WHERE session_id = ? AND role = 'user'
                            ORDER BY id DESC LIMIT 1
                        )
                        """,
                        (session.id,),
                    )
                session.messages.pop()
                self._cache[session.id] = session
            finally:
                conn.close()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._cache:
                return False
            conn = _get_conn()
            try:
                with conn:
                    # messages 表有 ON DELETE CASCADE，会自动级联删除
                    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                del self._cache[session_id]
            finally:
                conn.close()
        return True
