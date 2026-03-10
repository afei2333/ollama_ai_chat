"""
chat_manager.py — 对话管理核心
  ChatManager 是顶层业务协调器，整合 KnowledgeBase / SessionStore / ToolExecutor。
  自身不包含 I/O 或持久化细节，通过依赖注入便于单元测试。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, Iterator, List, Optional
from uuid import uuid4

import ollama
from ollama._types import ResponseError

from config import (
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    LLM_MAX_RETRIES,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    TOOL_DEFINITIONS,
)
from models import SessionState
from storage import KnowledgeBase, SessionStore
from tool_parser import extract_tool_calls_from_text, normalize_tool_call
from tools import ToolExecutor
from logger import get_logger, get_llm_logger

logger     = get_logger(__name__)
llm_logger = get_llm_logger()


class ChatManager:
    """
    顶层业务协调器。
    依赖注入 KnowledgeBase / SessionStore / ToolExecutor，
    三者均有默认实现，也可在测试中替换为 Mock。
    """

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        kb: Optional[KnowledgeBase] = None,
        store: Optional[SessionStore] = None,
        executor: Optional[ToolExecutor] = None,
    ) -> None:
        self.default_model     = default_model
        self.kb                = kb or KnowledgeBase()
        self._store            = store or SessionStore()
        self._executor         = executor or ToolExecutor()
        self.last_used_model: Optional[str] = None

    # -----------------------------------------------------------------------
    # 会话管理
    # -----------------------------------------------------------------------

    def create_session(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> SessionState:
        from storage import _now_iso  # 避免循环导入
        now = _now_iso()
        session = SessionState(
            id=uuid4().hex,
            name=(name or f"新对话 {now[-8:]}").strip(),
            model=(model or self.default_model).strip(),
            max_turns=max(1, min(max_turns, 50)),
            system_prompt=system_prompt.strip(),
            created_at=now,
            updated_at=now,
        )
        self._store.upsert(session)
        return session

    def list_sessions(self) -> List[SessionState]:
        return self._store.list_all()

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._store.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        ok = self._store.delete(session_id)
        if ok:
            logger.info("会话已删除 session=%s", session_id[:8])
        return ok

    def update_session(
        self,
        session_id: str,
        model: Optional[str] = None,
        max_turns: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Optional[SessionState]:
        from storage import _now_iso
        session = self._store.get(session_id)
        if not session:
            return None
        if model is not None:
            session.model = model.strip()
        if max_turns is not None:
            session.max_turns = max(1, min(max_turns, 50))
        if name and name.strip():
            session.name = name.strip()
        session.updated_at = _now_iso()
        self._store.upsert(session)
        return session

    # -----------------------------------------------------------------------
    # 上下文组装
    # -----------------------------------------------------------------------

    def _build_model_messages(
        self, session: SessionState, rag_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """组装发给模型的完整消息列表（system + RAG + 对话历史）。"""
        history_limit = session.max_turns * 2 + 5
        history = session.messages[-history_limit:]

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": session.system_prompt}]
        if rag_context:
            msgs.append({"role": "system", "content": rag_context})
        msgs.extend(history)
        return msgs

    def _build_rag_context(self, query: str, top_k: int) -> Optional[str]:
        hits = self.kb.search(query, top_k=max(1, min(top_k, 8)))
        if not hits:
            return None
        blocks = [
            f"[资料{i}] 标题: {doc.title}\n内容:\n{doc.content}"
            for i, (doc, _) in enumerate(hits, start=1)
        ]
        return "以下是可能相关的知识库资料：\n\n" + "\n\n".join(blocks)

    # -----------------------------------------------------------------------
    # 流式解析
    # -----------------------------------------------------------------------

    def _stream_tokens_and_calls(
        self, stream: Iterator
    ) -> Generator[Dict[str, Any], None, None]:
        """
        消费 Ollama 原始流：
          - 逐 token yield {"type": "text", "content": ...}
          - 流结束后若有工具调用，统一 yield {"type": "tool_call", ...}
        """
        full_content = ""
        tool_calls: List[Dict[str, Any]] = []

        for chunk in stream:
            msg = chunk.get("message", {})

            raw_tcs = msg.get("tool_calls") or []
            for raw_tc in raw_tcs:
                normalized = normalize_tool_call(raw_tc)
                if normalized:
                    tool_calls.append(normalized)
            if raw_tcs:
                continue

            content: str = msg.get("content", "")
            if content:
                full_content += content
                yield {"type": "text", "content": content}

        # Fallback：从纯文本中提取工具调用
        if not tool_calls and full_content:
            tool_calls = extract_tool_calls_from_text(full_content)

        if tool_calls:
            yield {"type": "tool_call", "tool_calls": tool_calls, "full_content": full_content}

    # -----------------------------------------------------------------------
    # 流式核心（含重试）
    # -----------------------------------------------------------------------

    def _stream_internal(
        self, session: SessionState, model_messages: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """核心流式请求，处理文本流与工具调用保存，支持 JSON 解析失败自动重试。"""
        self.last_used_model = session.model

        llm_logger.debug(
            "── LLM REQUEST ── session=%s model=%s messages=%d\n%s",
            session.id[:8], session.model, len(model_messages),
            json.dumps(model_messages, ensure_ascii=False, indent=2),
        )
        logger.info(
            "开始 LLM 请求 session=%s model=%s 消息数=%d",
            session.id[:8], session.model, len(model_messages),
        )

        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                stream = ollama.chat(
                    model=session.model,
                    messages=model_messages,
                    stream=True,
                    tools=TOOL_DEFINITIONS,
                    options={"num_ctx": LLM_NUM_CTX, "num_predict": LLM_NUM_PREDICT},
                )

                full_content       = ""
                final_tool_calls: List[Dict[str, Any]] = []

                for event in self._stream_tokens_and_calls(stream):
                    if event["type"] == "text":
                        full_content += event["content"]
                        yield event
                    elif event["type"] == "tool_call":
                        final_tool_calls = event["tool_calls"]
                        full_content     = event.get("full_content", full_content)

                self._log_llm_response(session, full_content, final_tool_calls)
                self._persist_response(session, full_content, final_tool_calls)

                if final_tool_calls:
                    yield {"type": "tool_call", "tool_calls": final_tool_calls}

                return  # 成功完成，跳出重试循环

            except Exception as exc:
                if self._is_json_error(exc) and attempt < LLM_MAX_RETRIES:
                    logger.warning(
                        "Ollama JSON 解析失败，第 %d/%d 次重试 session=%s: %s",
                        attempt + 1, LLM_MAX_RETRIES, session.id[:8], exc,
                    )
                    continue

                logger.error("LLM 请求异常 session=%s: %s", session.id[:8], exc, exc_info=True)
                self._store.pop_last_user_message(session)
                raise

    # -----------------------------------------------------------------------
    # 私有辅助
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_json_error(exc: Exception) -> bool:
        msg = str(exc)
        return "unexpected end of JSON" in msg or "failed to parse JSON" in msg

    def _log_llm_response(
        self,
        session: SessionState,
        full_content: str,
        tool_calls: List[Dict[str, Any]],
    ) -> None:
        if tool_calls:
            llm_logger.debug(
                "── LLM RESPONSE (tool_call) ── session=%s\ntext=%s\ntool_calls=%s",
                session.id[:8], full_content,
                json.dumps(tool_calls, ensure_ascii=False, indent=2),
            )
            logger.info(
                "LLM 触发工具调用 session=%s tools=%s",
                session.id[:8],
                [tc.get("function", {}).get("name") for tc in tool_calls],
            )
        else:
            llm_logger.debug(
                "── LLM RESPONSE (text) ── session=%s\n%s",
                session.id[:8], full_content,
            )
            logger.info("LLM 回复完成 session=%s 字符数=%d", session.id[:8], len(full_content))

    def _persist_response(
        self,
        session: SessionState,
        full_content: str,
        tool_calls: List[Dict[str, Any]],
    ) -> None:
        if tool_calls:
            self._store.append_message(
                session,
                {"role": "assistant", "content": full_content, "tool_calls": tool_calls},
            )
        elif full_content:
            self._store.append_message(session, {"role": "assistant", "content": full_content})

    # -----------------------------------------------------------------------
    # 公开对话接口
    # -----------------------------------------------------------------------

    def stream_chat(
        self,
        session: SessionState,
        user_message: str,
        top_k: int = 3,
        use_rag: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        logger.info(
            "用户消息 session=%s use_rag=%s 长度=%d",
            session.id[:8], use_rag, len(user_message),
        )
        rag_context = self._build_rag_context(user_message, top_k) if use_rag else None
        if rag_context:
            logger.debug("RAG 命中 session=%s top_k=%d", session.id[:8], top_k)

        self._store.append_message(session, {"role": "user", "content": user_message})
        model_messages = self._build_model_messages(session, rag_context=rag_context)
        yield from self._stream_internal(session, model_messages)

    def execute_and_stream(
        self, session: SessionState, tool_calls: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """执行经用户确认的工具调用，并将结果喂回模型继续生成。"""
        for tc in tool_calls:
            func      = tc.get("function", {})
            func_name = func.get("name", "")
            args      = func.get("arguments", {})
            output    = self._executor.execute(func_name, args)
            self._store.append_message(
                session, {"role": "tool", "name": func_name, "content": output}
            )

        model_messages = self._build_model_messages(session)
        yield from self._stream_internal(session, model_messages)

    # -----------------------------------------------------------------------
    # 模型管理
    # -----------------------------------------------------------------------

    def list_models(self) -> List[str]:
        try:
            resp  = ollama.list()
            names = [
                item.get("name") or item.get("model")
                for item in resp.get("models", [])
            ]
            unique = sorted({n for n in names if n})
            return unique or [self.default_model]
        except Exception:
            return [self.default_model]

    def unload_model(self, model: Optional[str] = None) -> None:
        target = model or self.last_used_model
        if not target:
            return
        try:
            ollama.generate(model=target, prompt="", keep_alive=0)
            logger.info("模型 %s 已卸载。", target)
        except Exception as exc:
            logger.warning("卸载模型 %s 失败: %s", target, exc)
