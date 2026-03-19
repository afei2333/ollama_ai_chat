"""
tools/executor.py — 工具执行器
  - ToolExecutor：注册模式，便于外部扩展新工具
  - 内置工具：run_shell / run_python / write_file / read_file / web_search
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, Optional

from tavily import TavilyClient

from config import TAVILY_API_KEY, TOOL_TIMEOUT
from core.logger import get_logger

logger = get_logger(__name__)

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
        self.register("run_shell",  self._handle_run_shell)
        self.register("run_python", self._handle_run_python)
        self.register("write_file", self._handle_write_file)
        self.register("read_file",  self._handle_read_file)
        self.register("web_search", self._handle_web_search)

    def register(self, name: str, handler: ToolHandler) -> None:
        """注册自定义工具，可在运行时动态扩展。"""
        self._handlers[name] = handler

    def execute(self, func_name: str, args: Dict[str, Any]) -> str:
        handler = self._handlers.get(func_name)
        if handler is None:
            logger.warning("调用了未注册的工具: %s", func_name)
            return f"[未知工具]: {func_name}"
        logger.info("执行工具 [%s] args=%s", func_name, json.dumps(args, ensure_ascii=False))
        result = handler(args)
        preview = result[:500] + "…" if len(result) > 500 else result
        logger.info("工具 [%s] 执行完毕，输出预览: %s", func_name, preview)
        return result

    # -----------------------------------------------------------------------
    # 内置工具实现
    # -----------------------------------------------------------------------

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

    @staticmethod
    def _handle_write_file(args: Dict[str, Any]) -> str:
        """将内容写入指定文件，支持相对路径和绝对路径，自动创建父目录。"""
        filename: str = args.get("filename", "").strip()
        content: str  = args.get("content", "")

        if not filename:
            return "[write_file 错误]: 文件名不能为空。"

        try:
            parent = os.path.dirname(filename)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)

            abs_path = os.path.abspath(filename)
            lines    = content.count("\n") + 1 if content else 0
            return (
                f"✅ 文件写入成功\n"
                f"路径: {abs_path}\n"
                f"大小: {len(content)} 字符 / {lines} 行"
            )
        except PermissionError:
            return f"[write_file 错误]: 无写入权限 — {filename}"
        except Exception as exc:
            return f"[write_file 错误]: {exc}"

    def _handle_web_search(self, args: Dict[str, Any]) -> str:
        """调用 Tavily 搜索，返回结构化摘要供模型使用。"""
        if self._tavily is None:
            return "[web_search 不可用]: 未配置 TAVILY_API_KEY，请先在 .env 中设置后重启服务。"

        query: str       = args.get("query", "").strip()
        max_results: int = min(int(args.get("max_results", 5)), 10)

        if not query:
            return "[web_search 错误]: 搜索关键词不能为空。"

        try:
            response = self._tavily.search(
                query=query,
                max_results=max_results,
                include_answer=True,
                include_raw_content=False,
            )
        except Exception as exc:
            return f"[web_search 请求失败]: {exc}"

        parts = []
        answer: str = response.get("answer", "").strip()
        if answer:
            parts.append(f"【摘要】{answer}")

        results = response.get("results", [])
        if not results:
            return answer or "[web_search]: 未找到相关结果。"

        for i, r in enumerate(results, start=1):
            parts.append(
                f"[结果{i}] {r.get('title', '无标题').strip()}\n"
                f"来源: {r.get('url', '')}  (相关度: {r.get('score', 0):.2f})\n"
                f"{r.get('content', '').strip()}"
            )

        return "\n\n".join(parts)

    @staticmethod
    def _handle_read_file(args: Dict[str, Any]) -> str:
        path: str = args.get("path", "").strip()

        if not path:
            return "[read_file 错误]: 文件路径不能为空。"
        if not os.path.isfile(path):
            return f"[read_file 错误]: 文件不存在 — {path}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            abs_path = os.path.abspath(path)
            lines    = content.count("\n") + 1 if content else 0
            return (
                f"✅ 文件读取成功\n"
                f"路径: {abs_path}\n"
                f"大小: {len(content)} 字符 / {lines} 行\n\n"
                f"{content}"
            )
        except PermissionError:
            return f"[read_file 错误]: 无读取权限 — {path}"
        except Exception as exc:
            return f"[read_file 错误]: {exc}"
