"""
config.py — 全局常量、工具定义、默认提示词
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 路径 & 基础常量
# ---------------------------------------------------------------------------

DATA_DIR      = "data"
KB_FILE       = os.path.join(DATA_DIR, "kb.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

DEFAULT_MODEL    = "qwen2.5:7b"
DEFAULT_MAX_TURNS = 8
TOOL_TIMEOUT     = 30       # 秒
LLM_NUM_CTX      = 8192     # 上下文窗口大小
LLM_NUM_PREDICT  = 2048     # 单次最大生成 token
LLM_MAX_RETRIES  = 2        # JSON 解析失败最大重试次数

# Tavily API Key
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# 默认系统提示词
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "你是一个强大的本地 AI 智能体助手。你有能力分析问题、拆解问题，解决问题。"
    "根据需求使用工具完成任务：\n"
    "- run_shell：执行本地终端命令，适合系统操作、文件管理等；\n"
    "- web_search：通过网络搜索获取最新信息，适合时事、技术文档、价格查询等；\n"
    "- read_file：读取本地文件内容；\n"
    "- write_file：将代码或文本内容写入到本地文件，当用户要求生成文件、保存代码时必须使用此工具。\n"
    "工具执行完毕后，基于输出结果向用户提供最终回答。\n"
    "重要：当用户要求将代码写入文件时，必须调用 write_file 工具，不能只在对话中展示代码。"
)

# ---------------------------------------------------------------------------
# 工具定义（发送给 Ollama 的 JSON Schema）
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    # 工具定义示例
    # 执行终端命令
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
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将代码或文本内容写入到指定文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "要写入的文件名，例如 main.py",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入文件的完整内容",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "读取本地文件的完整内容并返回。适用于查看代码、配置、日志等文本文件。"
            "文件内容会直接返回，无需在参数中填写内容。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径，支持相对路径和绝对路径，例如 ./main.py",
                },
            },
            "required": ["path"],
        },
    },
},
]
