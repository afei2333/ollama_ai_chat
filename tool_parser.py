"""
tool_parser.py — 工具调用解析工具函数
  - _normalize_tool_call      : 将 Ollama 返回的 ToolCall 对象统一为内部 dict 格式
  - _extract_tool_calls_from_text : Fallback，从纯文本中提取 JSON 工具调用
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def normalize_tool_call(tc: Any) -> Optional[Dict[str, Any]]:
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


def extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Fallback：当模型未通过原生 tool_calls 字段返回工具调用时，
    尝试从输出文本中解析形如 {"name": ..., "arguments": ...} 的 JSON 片段。
    """
    tool_calls: List[Dict[str, Any]] = []

    for match in re.finditer(r'\{\s*"name"', text):
        brace_count = 0
        json_chars: List[str] = []

        for char in text[match.start():]:
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
                tool_calls.append({
                    "type": "function",
                    "function": {
                        "name": data["name"],
                        "arguments": data["arguments"],
                    },
                })
        except json.JSONDecodeError:
            pass

    return tool_calls
