"""
config.py — 全局常量、工具定义、默认提示词（ReAct 模式）
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

DEFAULT_MODEL    = "qwen3.5:9b"
DEFAULT_MAX_TURNS = 8
TOOL_TIMEOUT     = 30       # 秒
LLM_NUM_CTX      = 8192     # 上下文窗口大小
LLM_NUM_PREDICT  = 2048     # 单次最大生成 token
LLM_MAX_RETRIES  = 2        # JSON 解析失败最大重试次数

# ReAct 模式配置
REACT_MAX_STEPS  = 10       # 单次任务最大 Thought→Action→Observation 循环步数
REACT_STOP_TOKEN = "Final Answer:"  # 模型输出此前缀时终止循环

# Tavily API Key
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Google Gemini 配置
# ---------------------------------------------------------------------------

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_HTTP_PROXY: str = os.getenv("GOOGLE_HTTP_PROXY", "")  # 可选代理，如 http://127.0.0.1:10808

# Google 模型列表（前缀 "google:" 标识，用于前端/后端路由区分）
GOOGLE_MODELS: List[str] = [
    "google:gemini-3-flash-preview",
    "google:gemini-3.1-pro-preview",
]

def is_google_model(model: str) -> bool:
    """判断是否为 Google 模型（以 'google:' 开头）。"""
    return model.startswith("google:")

# ---------------------------------------------------------------------------
# ReAct 系统提示词
# ---------------------------------------------------------------------------
#
# ReAct 模式要求模型严格按照以下循环格式输出：
#
#   Thought: <对当前状态的推理>
#   Action: <工具名称>
#   Action Input: <工具参数，JSON 格式>
#   Observation: <工具返回结果（由系统填充）>
#   ... （可循环多次）
#   Thought: 我已经知道最终答案了。
#   Final Answer: <最终回复用户的内容>
#
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = (
    "你是一个强大的 AI 智能体，使用 ReAct（Reasoning + Acting）框架逐步解决问题。\n\n"
    "工作路径：'E:/code/ollama_ai_chat/workspace'，你的操作权限仅限于此路径，禁止操作其他路径。\n\n"
    "## 可用工具\n"
    "- run_shell：执行本地终端命令，适合系统操作、文件管理等\n"
    "- web_search：通过网络搜索获取最新信息，适合时事、技术文档、价格查询等\n"
    "- read_file：读取本地文件内容\n"
    "- write_file：将代码或文本内容写入本地文件\n\n"
    "## 输出格式（必须严格遵守）\n\n"
    "每一步按以下格式输出，不得跳过任何字段：\n\n"
    "Thought: <分析当前情况，判断下一步需要做什么>\n"
    "Action: <工具名称，必须是上方列表之一>\n"
    "Action Input: <工具所需参数，严格 JSON 格式>\n\n"
    "系统执行工具后会自动追加：\n"
    "Observation: <工具执行结果>\n\n"
    "然后继续下一轮 Thought/Action/Observation，直到问题解决后输出：\n\n"
    "Thought: 我已经知道最终答案了。\n"
    "Final Answer: <完整、清晰的最终回复>\n\n"
    "## 规则\n"
    "1. 每次只能调用一个工具，等待 Observation 后再决定下一步。\n"
    "2. 当用户要求生成文件时，必须调用 write_file，不能只在对话中展示代码。\n"
    "3. 若用户要求操作工作路径之外的文件，立即在 Final Answer 中拒绝。\n"
    "4. 若工具执行出错，在 Thought 中分析原因并尝试修正。\n"
    "5. 不要编造工具结果，始终等待真实 Observation。\n"
)

# 兼容旧版：保留 DEFAULT_SYSTEM_PROMPT 指向 ReAct 提示词
DEFAULT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# ReAct 解析辅助函数
# ---------------------------------------------------------------------------

import re
import json


def parse_react_step(text: str) -> Dict[str, Any]:
    """
    从模型输出文本中解析一个 ReAct 步骤。

    返回字典，可能包含以下键：
      - thought (str)
      - action (str)
      - action_input (dict | str)
      - final_answer (str)

    示例输入：
        Thought: 需要查看当前目录结构。
        Action: run_shell
        Action Input: {"command": "dir"}

    示例输出：
        {"thought": "需要查看当前目录结构。",
         "action": "run_shell",
         "action_input": {"command": "dir"}}
    """
    result: Dict[str, Any] = {}

    # 提取 Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # 检测 Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    # 提取 Action
    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # 提取 Action Input（JSON）
    input_match = re.search(r"Action Input:\s*(\{.*?\}|\[.*?\]|\".*?\"|.+?)(?=\nObservation:|\nThought:|$)", text, re.DOTALL)
    if input_match:
        raw = input_match.group(1).strip()
        try:
            result["action_input"] = json.loads(raw)
        except json.JSONDecodeError:
            result["action_input"] = raw  # 回退为原始字符串

    return result


def format_observation(tool_name: str, output: str) -> str:
    """将工具执行结果格式化为 ReAct Observation 行。"""
    return f"Observation: [{tool_name}] {output.strip()}"


def is_final_answer(text: str) -> bool:
    """判断模型输出是否包含终止信号。"""
    return REACT_STOP_TOKEN in text


# ---------------------------------------------------------------------------
# 工具定义（发送给 Ollama 的 JSON Schema）
# ---------------------------------------------------------------------------

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