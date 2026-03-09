"""
ollama_client.py — 本地 Ollama AI 助手核心逻辑
重构要点:
  - 将 KnowledgeBase / SessionStore / ToolExecutor / ChatManager 职责分离
  - 消除重复的 subprocess import
  - 将魔法数字、路径、默认值统一提取为常量
  - ToolExecutor 支持注册式扩展，不再用 if/elif 硬编码
  - _stream_internal 拆分为更小的子方法，降低圈复杂度
  - 类型标注更完整，兼容 Python 3.10+
  - 新增 web_search 工具，基于 Tavily API
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple
from uuid import uuid4

import ollama
from tavily import TavilyClient
from dotenv import load_dotenv

from logger import get_logger, get_llm_logger

load_dotenv()  # 自动读取项目根目录的 .env 文件

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

DATA_DIR = "data"
KB_FILE = os.path.join(DATA_DIR, "kb.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_MAX_TURNS = 8
TOOL_TIMEOUT = 30  # 秒

# Tavily API Key —— 优先从环境变量读取，也可在 .env 或启动脚本中设置
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

DEFAULT_SYSTEM_PROMPT = (
    "你是一个强大的本地 AI 智能体助手。你有能力分析并拆解问题，"
    "根据需求使用工具完成任务：\n"
    "- run_shell：执行本地终端命令，适合系统操作、文件管理等；\n"
    "- web_search：通过网络搜索获取最新信息，适合时事、技术文档、价格查询等。\n"
    "工具执行完毕后，基于输出结果向用户提供最终回答。"
)

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "在本地机器的命令行终端（Shell/CMD）中执行系统命令并返回输出结果。"
                "适用于系统状态查询、文件管理、网络连通性测试等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的终端命令，例如 dir, ls, ping, ipconfig 等。",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "通过 Tavily 搜索引擎从互联网获取最新信息。"
                "适用于：实时新闻、当前价格、最新技术文档、天气、赛事结果等"
                "本地知识库无法覆盖的内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题，建议使用英文以获得更好结果。",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果条数，默认 5，最多 10。",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

logger     = get_logger(__name__)
llm_logger = get_llm_logger()
os.makedirs(DATA_DIR, exist_ok=True)


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
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeDoc:
    id: str
    title: str
    content: str
    tokens: set
    created_at: str


@dataclass
class SessionState:
    id: str
    name: str
    model: str
    max_turns: int
    system_prompt: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """线程安全的轻量知识库，基于关键词 Token 匹配做 RAG 检索。"""

    def __init__(self) -> None:
        self._docs: Dict[str, KnowledgeDoc] = {}
        self._lock = threading.Lock()
        self._load()

    # --- 持久化 ---

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

    # --- 公开接口 ---

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
# 会话存储
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

    def save_session(self, session: SessionState) -> None:
        """持久化单个会话（调用方负责持有或不持有锁）。"""
        with self._lock:
            self._sessions[session.id] = session
            self._save()

    def upsert(self, session: SessionState) -> None:
        self.save_session(session)

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
        """删除会话，返回是否成功。"""
        with self._lock:
            removed = self._sessions.pop(session_id, None) is not None
            if removed:
                self._save()
        return removed


# ---------------------------------------------------------------------------
# 工具执行器（注册模式，便于扩展）
# ---------------------------------------------------------------------------

ToolHandler = Callable[[Dict[str, Any]], str]


class ToolExecutor:
    """可注册任意工具处理函数的执行器，解耦工具逻辑与对话逻辑。"""

    def __init__(self) -> None:
        self._handlers: Dict[str, ToolHandler] = {}
        self._tavily: Optional[TavilyClient] = (
            TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
        )
        self._register_builtin()

    def _register_builtin(self) -> None:
        self.register("run_shell", self._handle_run_shell)
        self.register("run_python", self._handle_run_python)
        self.register("web_search", self._handle_web_search)

    def register(self, name: str, handler: ToolHandler) -> None:
        self._handlers[name] = handler

    def execute(self, func_name: str, args: Dict[str, Any]) -> str:
        handler = self._handlers.get(func_name)
        if handler is None:
            logger.warning("调用了未注册的工具: %s", func_name)
            return f"[未知工具]: {func_name}"
        logger.info("执行工具 [%s] args=%s", func_name, json.dumps(args, ensure_ascii=False))
        result = handler(args)
        # 截断过长输出，避免日志膨胀（完整内容已存入会话消息）
        preview = result[:500] + "…" if len(result) > 500 else result
        logger.info("工具 [%s] 执行完毕，输出预览: %s", func_name, preview)
        return result

    # --- 内置工具实现 ---

    @staticmethod
    def _run_subprocess(cmd: Any, shell: bool = False) -> str:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=shell,
                timeout=TOOL_TIMEOUT,
            )
            output = result.stdout + result.stderr
            return output.strip() or "[命令执行成功，但没有任何输出内容。]"
        except subprocess.TimeoutExpired:
            return f"[执行失败]: 命令执行超过 {TOOL_TIMEOUT} 秒超时。"
        except Exception as exc:
            return f"[执行出错]: {exc}"

    def _handle_run_shell(self, args: Dict[str, Any]) -> str:
        command = args.get("command", "")
        return self._run_subprocess(command, shell=True)

    def _handle_run_python(self, args: Dict[str, Any]) -> str:
        code = args.get("code", "")
        return self._run_subprocess([sys.executable, "-c", code], shell=False)

    def _handle_web_search(self, args: Dict[str, Any]) -> str:
        """调用 Tavily 搜索，返回结构化摘要供模型使用。"""
        if self._tavily is None:
            return "[web_search 不可用]: 未配置 TAVILY_API_KEY 环境变量，请先设置后重启服务。"

        query: str = args.get("query", "").strip()
        max_results: int = min(int(args.get("max_results", 5)), 10)

        if not query:
            return "[web_search 错误]: 搜索关键词不能为空。"

        try:
            response = self._tavily.search(
                query=query,
                max_results=max_results,
                include_answer=True,       # 让 Tavily 额外返回一句话摘要
                include_raw_content=False, # 不需要原始 HTML，减少 token 消耗
            )
        except Exception as exc:
            return f"[web_search 请求失败]: {exc}"

        parts: List[str] = []

        # Tavily 自带的一句话 AI 摘要（可能为空）
        answer: str = response.get("answer", "").strip()
        if answer:
            parts.append(f"【摘要】{answer}")

        # 逐条结果
        results: List[Dict[str, Any]] = response.get("results", [])
        if not results:
            return answer or "[web_search]: 未找到相关结果。"

        for i, r in enumerate(results, start=1):
            title   = r.get("title", "无标题").strip()
            url     = r.get("url", "")
            content = r.get("content", "").strip()
            score   = r.get("score", 0)
            parts.append(
                f"[结果{i}] {title}\n来源: {url}  (相关度: {score:.2f})\n{content}"
            )

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 工具调用解析
# ---------------------------------------------------------------------------

def _normalize_tool_call(tc: Any) -> Optional[Dict[str, Any]]:
    """统一将 Ollama 返回的 ToolCall 对象或 dict 规范化为内部格式。"""
    if isinstance(tc, dict):
        return tc
    try:
        return {
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": dict(tc.function.arguments),
            },
        }
    except Exception:
        return None


def _extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Fallback：当模型未通过原生 tool_calls 字段返回工具调用时，
    尝试从输出文本中解析形如 {"name": ..., "arguments": ...} 的 JSON 片段。
    """
    tool_calls: List[Dict[str, Any]] = []
    for start in (m.start() for m in re.finditer(r'\{\s*"name"', text)):
        brace_count = 0
        json_chars: List[str] = []
        for char in text[start:]:
            json_chars.append(char)
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
        try:
            data = json.loads("".join(json_chars))
            if "name" in data and "arguments" in data:
                tool_calls.append(
                    {"type": "function", "function": {"name": data["name"], "arguments": data["arguments"]}}
                )
        except json.JSONDecodeError:
            pass
    return tool_calls


# ---------------------------------------------------------------------------
# ChatManager — 对话管理（依赖注入，便于测试）
# ---------------------------------------------------------------------------

class ChatManager:
    """
    顶层业务协调器，整合 KnowledgeBase / SessionStore / ToolExecutor。
    自身不再包含 I/O 或持久化细节。
    """

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        kb: Optional[KnowledgeBase] = None,
        store: Optional[SessionStore] = None,
        executor: Optional[ToolExecutor] = None,
    ) -> None:
        self.default_model = default_model
        self.kb = kb or KnowledgeBase()
        self._store = store or SessionStore()
        self._executor = executor or ToolExecutor()
        self.last_used_model: Optional[str] = None

    # --- 会话管理 ---

    def create_session(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
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

    # --- 上下文组装 ---

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

    # --- 流式对话核心 ---

    def _collect_stream(
        self, stream: Iterator
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        消费 Ollama 流，返回 (full_text, tool_calls)。
        文本 token 通过 generator.send() 不适用此处，故分两步：
        先收集，再由调用方 yield token。
        """
        # 注意：此处边收集边 yield 需要重构为生成器包装
        # 见 _stream_tokens_and_calls()
        raise NotImplementedError("请使用 _stream_tokens_and_calls")

    def _stream_tokens_and_calls(
        self, stream: Iterator
    ) -> Generator[Dict[str, Any], None, None]:
        """
        消费 Ollama 原始流，逐 token yield {"type":"text","content":...}，
        并在流结束后统一 yield {"type":"tool_call","tool_calls":[...]}（若有）。
        """
        full_content = ""
        tool_calls: List[Dict[str, Any]] = []

        for chunk in stream:
            msg = chunk.get("message", {})

            # 处理工具调用块
            raw_tcs = msg.get("tool_calls") or []
            for raw_tc in raw_tcs:
                normalized = _normalize_tool_call(raw_tc)
                if normalized:
                    tool_calls.append(normalized)
            if raw_tcs:
                continue

            # 处理文本块
            content: str = msg.get("content", "")
            if content:
                full_content += content
                yield {"type": "text", "content": content}

        # Fallback：从文本中解析工具调用
        if not tool_calls and full_content:
            tool_calls = _extract_tool_calls_from_text(full_content)

        if tool_calls:
            yield {"type": "tool_call", "tool_calls": tool_calls, "full_content": full_content}

    def _stream_internal(
        self, session: SessionState, model_messages: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """核心流式请求，处理文本流与工具调用保存。"""
        self.last_used_model = session.model

        # ── LLM 请求日志 ────────────────────────────────────────────────────
        llm_logger.debug(
            "── LLM REQUEST ── session=%s model=%s messages=%d\n%s",
            session.id[:8],
            session.model,
            len(model_messages),
            json.dumps(model_messages, ensure_ascii=False, indent=2),
        )
        logger.info(
            "开始 LLM 请求 session=%s model=%s 消息数=%d",
            session.id[:8], session.model, len(model_messages),
        )

        try:
            stream = ollama.chat(
                model=session.model,
                messages=model_messages,
                stream=True,
                tools=TOOL_DEFINITIONS,
            )

            full_content = ""
            final_tool_calls: List[Dict[str, Any]] = []

            for event in self._stream_tokens_and_calls(stream):
                if event["type"] == "text":
                    full_content += event["content"]
                    yield event
                elif event["type"] == "tool_call":
                    final_tool_calls = event["tool_calls"]
                    full_content = event.get("full_content", full_content)

            # ── LLM 响应日志 ─────────────────────────────────────────────────
            if final_tool_calls:
                llm_logger.debug(
                    "── LLM RESPONSE (tool_call) ── session=%s\ntext=%s\ntool_calls=%s",
                    session.id[:8],
                    full_content,
                    json.dumps(final_tool_calls, ensure_ascii=False, indent=2),
                )
                logger.info(
                    "LLM 触发工具调用 session=%s tools=%s",
                    session.id[:8],
                    [tc.get("function", {}).get("name") for tc in final_tool_calls],
                )
            else:
                llm_logger.debug(
                    "── LLM RESPONSE (text) ── session=%s\n%s",
                    session.id[:8],
                    full_content,
                )
                logger.info(
                    "LLM 回复完成 session=%s 字符数=%d",
                    session.id[:8], len(full_content),
                )

            # 持久化
            if final_tool_calls:
                self._store.append_message(
                    session,
                    {"role": "assistant", "content": full_content, "tool_calls": final_tool_calls},
                )
                yield {"type": "tool_call", "tool_calls": final_tool_calls}
            elif full_content:
                self._store.append_message(
                    session, {"role": "assistant", "content": full_content}
                )

        except Exception as exc:
            logger.error("LLM 请求异常 session=%s: %s", session.id[:8], exc, exc_info=True)
            self._store.pop_last_user_message(session)
            raise exc

    # --- 公开对话接口 ---

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
            func = tc.get("function", {})
            func_name: str = func.get("name", "")
            args: Dict[str, Any] = func.get("arguments", {})
            output = self._executor.execute(func_name, args)
            self._store.append_message(
                session, {"role": "tool", "name": func_name, "content": output}
            )

        model_messages = self._build_model_messages(session)
        yield from self._stream_internal(session, model_messages)

    # --- 模型管理 ---

    def list_models(self) -> List[str]:
        try:
            resp = ollama.list()
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