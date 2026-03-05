from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple
from uuid import uuid4

import ollama

DEFAULT_SYSTEM_PROMPT = (
    "你是一个本地 AI 助手。回答准确、简洁、有条理。"
    "当你不确定时，请明确说明不确定性并给出下一步建议。"
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())

@dataclass
class KnowledgeDoc:
    id: str
    title: str
    content: str
    tokens: set
    created_at: str

class KnowledgeBase:
    def __init__(self) -> None:
        self._docs: Dict[str, KnowledgeDoc] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        path = os.path.join(DATA_DIR, "kb.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        v["tokens"] = set(v.get("tokens", []))
                        self._docs[k] = KnowledgeDoc(**v)
            except Exception as e:
                print(f"Failed to load KB: {e}")

    def _save(self):
        path = os.path.join(DATA_DIR, "kb.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                data = {k: {**v.__dict__, "tokens": list(v.tokens)} for k, v in self._docs.items()}
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save KB: {e}")

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
            res = self._docs.pop(doc_id, None) is not None
            if res:
                self._save()
            return res

    def search(self, query: str, top_k: int = 3) -> List[Tuple[KnowledgeDoc, int]]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        with self._lock:
            docs = list(self._docs.values())

        scored: List[Tuple[KnowledgeDoc, int]] = []
        for doc in docs:
            score = len(query_tokens & doc.tokens)
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

@dataclass
class SessionState:
    id: str
    name: str
    model: str
    max_turns: int
    system_prompt: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, str]] = field(default_factory=list)

class ChatManager:
    def __init__(self, default_model: str = "qwen2.5:7b") -> None:
        self.default_model = default_model
        self.kb = KnowledgeBase()
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        path = os.path.join(DATA_DIR, "sessions.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self._sessions[k] = SessionState(**v)
            except Exception as e:
                print(f"Failed to load sessions: {e}")

    def _save(self):
        path = os.path.join(DATA_DIR, "sessions.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                data = {k: v.__dict__ for k, v in self._sessions.items()}
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save sessions: {e}")

    def create_session(
        self,
        name: str | None = None,
        model: str | None = None,
        max_turns: int = 8,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> SessionState:
        now = _now_iso()
        session = SessionState(
            id=uuid4().hex,
            name=(name or f"新对话 {now[-8:]}").strip(),
            model=(model or self.default_model).strip(),
            max_turns=max(1, min(max_turns, 50)),
            system_prompt=system_prompt.strip(),
            created_at=now,
            updated_at=now,
            messages=[],
        )
        with self._lock:
            self._sessions[session.id] = session
            self._save()
        return session

    def list_sessions(self) -> List[SessionState]:
        with self._lock:
            sessions = list(self._sessions.values())
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def get_session(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def update_session(
        self,
        session_id: str,
        model: str | None = None,
        max_turns: int | None = None,
        name: str | None = None,
    ) -> SessionState | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            if model is not None:
                session.model = model.strip()
            if max_turns is not None:
                session.max_turns = max(1, min(max_turns, 50))
            if name is not None and name.strip():
                session.name = name.strip()
            session.updated_at = _now_iso()
            self._save()
            return session

    def add_message(self, session: SessionState, role: str, content: str) -> None:
        with self._lock:
            session.messages.append({"role": role, "content": content})
            session.updated_at = _now_iso()
            self._save()

    def messages_for_model(
        self,
        session: SessionState,
        new_user_message: str,
        rag_context: str | None = None,
    ) -> List[Dict[str, str]]:
        history_limit = session.max_turns * 2
        history = session.messages[-history_limit:] if history_limit > 0 else session.messages

        msgs: List[Dict[str, str]] = [{"role": "system", "content": session.system_prompt}]
        if rag_context:
            msgs.append({"role": "system", "content": rag_context})
        msgs.extend(history)
        msgs.append({"role": "user", "content": new_user_message})
        return msgs

    def stream_chat(
        self,
        session: SessionState,
        user_message: str,
        top_k: int = 3,
        use_rag: bool = True,
    ):
        rag_context = None
        if use_rag:
            hits = self.kb.search(user_message, top_k=max(1, min(top_k, 8)))
            if hits:
                blocks = []
                for idx, (doc, score) in enumerate(hits, start=1):
                    blocks.append(
                        f"[资料{idx}] 标题: {doc.title}\n相关分: {score}\n内容:\n{doc.content}"
                    )
                rag_context = (
                    "以下是可能相关的知识库资料，仅在相关时引用，不要编造来源：\n\n"
                    + "\n\n".join(blocks)
                )

        model_messages = self.messages_for_model(session, user_message, rag_context=rag_context)
        self.add_message(session, "user", user_message)

        full = ""
        try:
            stream = ollama.chat(
                model=session.model,
                messages=model_messages,
                stream=True,
            )

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if not content:
                    continue
                full += content
                yield content
        finally:
            # 完整保存生成的回复，避免中断产生格式错乱导致后续模型报错退出
            if full:
                self.add_message(session, "assistant", full)
            else:
                # 若没有任何回复（异常直接中断），撤销最后一条 user 消息
                with self._lock:
                    if session.messages and session.messages[-1]["role"] == "user":
                        session.messages.pop()
                        self._save()

    def list_models(self) -> List[str]:
        try:
            resp = ollama.list()
            models = resp.get("models", [])
            names = []
            for item in models:
                name = item.get("name") or item.get("model")
                if name:
                    names.append(name)
            unique = sorted(set(names))
            return unique if unique else [self.default_model]
        except Exception:
            return [self.default_model]