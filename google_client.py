"""
google_client.py — Google Gemini API 流式对话适配器

代理必须在 import google 之前通过 os.environ 设置，SDK 初始化时读取。
"""

from __future__ import annotations

import os

# !! 代理必须在 import google 之前设置 !!
from config import GOOGLE_API_KEY, GOOGLE_HTTP_PROXY, TOOL_DEFINITIONS
from logger import get_logger, get_llm_logger

logger     = get_logger(__name__)
llm_logger = get_llm_logger()

if GOOGLE_HTTP_PROXY:
    os.environ["HTTP_PROXY"]  = GOOGLE_HTTP_PROXY
    os.environ["HTTPS_PROXY"] = GOOGLE_HTTP_PROXY
    logger.info("Google 代理已设置: %s", GOOGLE_HTTP_PROXY)

# 代理设置完成后再 import google
try:
    from google import genai
    from google.genai import types as gtypes
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google-genai 未安装，请执行: pip install google-genai")

from typing import Any, Dict, Generator, List, Optional


# ---------------------------------------------------------------------------
# 消息格式转换
# ---------------------------------------------------------------------------

def _tool_defs_to_gemini() -> List[Any]:
    """将 TOOL_DEFINITIONS（OpenAI 格式）转换为 Gemini FunctionDeclaration。"""
    if not _GENAI_AVAILABLE:
        return []
    declarations = []
    for td in TOOL_DEFINITIONS:
        fn = td.get("function", {})
        params_schema = fn.get("parameters", {})
        cleaned_props = {}
        for k, v in params_schema.get("properties", {}).items():
            cleaned_props[k] = {kk: vv for kk, vv in v.items() if kk != "default"}
        cleaned_schema = {
            "type": params_schema.get("type", "object"),
            "properties": cleaned_props,
        }
        if "required" in params_schema:
            cleaned_schema["required"] = params_schema["required"]
        declarations.append(
            gtypes.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=cleaned_schema,
            )
        )
    return declarations


def _ollama_messages_to_gemini(
    messages: List[Dict[str, Any]]
) -> tuple[Optional[str], List[Any]]:
    """将 Ollama 格式消息列表转换为 (system_instruction, gemini_contents)。"""
    if not _GENAI_AVAILABLE:
        return None, []

    system_parts: List[str] = []
    contents: List[Any] = []

    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            system_parts.append(content)

        elif role == "user":
            contents.append(gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=content)],
            ))

        elif role == "assistant":
            # 优先使用保存的原始 Content（含 thought_signature），避免 400 INVALID_ARGUMENT
            # 存储的是 dict（JSON 序列化后），需要先反序列化为 Content 对象
            raw_data = msg.get("_google_raw_content")
            raw = deserialize_google_content(raw_data) if raw_data is not None else None
            if raw is not None:
                contents.append(raw)
            else:
                parts: List[Any] = []
                if content:
                    parts.append(gtypes.Part(text=content))
                for tc in msg.get("tool_calls", []):
                    fn   = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    parts.append(gtypes.Part(
                        function_call=gtypes.FunctionCall(name=name, args=args)
                    ))
                if parts:
                    contents.append(gtypes.Content(role="model", parts=parts))

        elif role == "tool":
            func_name = msg.get("name", "tool")
            response_dict = {"output": content} if isinstance(content, str) else content
            contents.append(gtypes.Content(
                role="user",
                parts=[gtypes.Part(
                    function_response=gtypes.FunctionResponse(
                        name=func_name,
                        response=response_dict,
                    )
                )],
            ))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


# ---------------------------------------------------------------------------
# Content 序列化 / 反序列化（用于 JSON 持久化）
# ---------------------------------------------------------------------------

def serialize_google_content(content_obj: Any) -> Optional[Dict]:
    """将 Gemini Content 对象转为可 JSON 序列化的 dict，保留 thought_signature。"""
    if content_obj is None or not _GENAI_AVAILABLE:
        return None
    try:
        # SDK 提供 _to_dict / model_dump 等方法
        if hasattr(content_obj, "model_dump"):
            return content_obj.model_dump(exclude_none=True)
        if hasattr(content_obj, "_to_dict"):
            return content_obj._to_dict()
        # 兜底：手动序列化 parts
        parts_data = []
        for part in (content_obj.parts or []):
            p: Dict[str, Any] = {}
            if hasattr(part, "text") and part.text:
                p["text"] = part.text
            if hasattr(part, "thought") and part.thought:
                p["thought"] = part.thought
            if hasattr(part, "thought_signature") and part.thought_signature:
                # thought_signature 是 bytes，转 base64 存储
                import base64
                sig = part.thought_signature
                p["thought_signature"] = base64.b64encode(sig).decode() if isinstance(sig, bytes) else sig
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                p["function_call"] = {"name": fc.name, "args": dict(fc.args or {})}
            parts_data.append(p)
        return {"role": content_obj.role, "parts": parts_data, "_raw_serialized": True}
    except Exception as e:
        return None


def deserialize_google_content(data: Optional[Dict]) -> Optional[Any]:
    """将持久化的 dict 还原为 Gemini Content 对象。"""
    if data is None or not _GENAI_AVAILABLE:
        return None
    try:
        if not data.get("_raw_serialized"):
            # 使用 SDK 标准反序列化
            return gtypes.Content.model_validate(data)
        # 手动构建 parts
        import base64
        parts = []
        for p in data.get("parts", []):
            kwargs: Dict[str, Any] = {}
            if "text" in p:
                kwargs["text"] = p["text"]
            if "thought" in p:
                kwargs["thought"] = p["thought"]
            if "thought_signature" in p:
                sig = p["thought_signature"]
                kwargs["thought_signature"] = base64.b64decode(sig) if isinstance(sig, str) else sig
            if "function_call" in p:
                fc = p["function_call"]
                kwargs["function_call"] = gtypes.FunctionCall(name=fc["name"], args=fc.get("args", {}))
            parts.append(gtypes.Part(**kwargs))
        return gtypes.Content(role=data.get("role", "model"), parts=parts)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 主流式调用函数
# ---------------------------------------------------------------------------

def stream_google(
    model_name: str,
    messages: List[Dict[str, Any]],
) -> Generator[Dict[str, Any], None, None]:
    """
    调用 Google Gemini API，流式 yield 与 ChatManager 兼容的事件：
      {"type": "text",      "content": "..."}
      {"type": "tool_call", "tool_calls": [...], "full_content": "..."}
    """
    if not _GENAI_AVAILABLE:
        yield {"type": "text", "content": "[错误] google-genai 未安装，请执行: pip install google-genai"}
        return

    if not GOOGLE_API_KEY:
        yield {"type": "text", "content": "[错误] 未配置 GOOGLE_API_KEY，请在 .env 中设置后重启服务。"}
        return

    real_model = model_name.removeprefix("google:")
    system_instruction, contents = _ollama_messages_to_gemini(messages)

    llm_logger.debug("── GOOGLE REQUEST ── model=%s messages=%d", real_model, len(messages))
    logger.info("开始 Google Gemini 请求 model=%s 消息数=%d", real_model, len(messages))

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        tool_declarations = _tool_defs_to_gemini()
        tools_param = [gtypes.Tool(function_declarations=tool_declarations)] if tool_declarations else []

        config = gtypes.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools_param,
        )

        full_text   = ""
        tool_calls: List[Dict[str, Any]] = []
        # 保存完整的原始 Content 对象（含 thought_signature），用于回传给 API
        raw_model_content: Optional[Any] = None

        response_stream = client.models.generate_content_stream(
            model=real_model,
            contents=contents,
            config=config,
        )

        # 流式收集所有 chunk，合并为最终 Content
        all_parts: List[Any] = []
        for chunk in response_stream:
            try:
                text_part = chunk.text
            except Exception:
                text_part = None

            if text_part:
                full_text += text_part
                yield {"type": "text", "content": text_part}

            try:
                for candidate in (chunk.candidates or []):
                    for part in (candidate.content.parts or []):
                        all_parts.append(part)
                        if part.function_call:
                            fc = part.function_call
                            tool_calls.append({
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": dict(fc.args) if fc.args else {},
                                },
                            })
            except Exception:
                pass

        # 构建完整的原始 Content，保留 thought_signature 等隐藏字段
        if all_parts:
            raw_model_content = gtypes.Content(role="model", parts=all_parts)

        if tool_calls:
            logger.info("Google Gemini 触发工具调用: %s", [tc["function"]["name"] for tc in tool_calls])
            yield {
                "type": "tool_call",
                "tool_calls": tool_calls,
                "full_content": full_text,
                "_google_raw_content": raw_model_content,  # 原始 Content，含 thought_signature
            }
        else:
            logger.info("Google Gemini 回复完成 字符数=%d", len(full_text))

    except Exception as exc:
        logger.error("Google Gemini 请求异常: %s", exc, exc_info=True)
        raise