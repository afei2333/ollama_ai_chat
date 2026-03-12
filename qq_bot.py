"""
qq_bot.py — 多账号 QQ 机器人主程序

支持同时运行任意数量的 QQ Bot，每个账号独立配置。
共用同一个 ChatManager 实例（共享知识库和工具）。

内置指令（私聊直接发，群聊 @机器人 后发）：
  /stop  — 立即中断当前正在生成的回复
  /new   — 清除当前对话记忆，开启新对话

配置文件 bots.json：
  [
    {
      "app_id":      "AppID_1",
      "app_secret":  "AppSecret_1",
      "name":        "客服Bot",
      "model":       "qwen3.5:9b",
      "sandbox":     false,
      "max_length":  1500,
      "system_prompt": "你是..."
    }
  ]

启动：
    python qq_bot.py
    python qq_bot.py --config /path/to/bots.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from config import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT
from chat_manager import ChatManager
from qqbot_client import QQBotClient
from logger import get_logger

log = get_logger(__name__)

# 所有 Bot 实例共用同一个 ChatManager（共享知识库 / 工具 / SQLite）
_shared_manager = ChatManager(default_model=DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# BotInstance — 封装单个账号的全部状态与逻辑
# ---------------------------------------------------------------------------

class BotInstance:
    """
    一个 BotInstance 对应一个腾讯开放平台应用（AppID）。

    中断机制说明：
      - _sync_generate 运行在线程池（run_in_executor），无法直接取消。
      - 通过 threading.Event（stop_event）在生成循环的每个 token 之间检查
        中断信号，收到信号后提前返回已生成的内容。
      - asyncio.Task.cancel() 将 stop_event 置位，确保线程侧也能感知。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.app_id       : str  = str(cfg["app_id"]).strip()
        self.app_secret   : str  = str(cfg["app_secret"]).strip()
        self.name         : str  = cfg.get("name") or f"Bot_{self.app_id[:8]}"
        self.model        : str  = cfg.get("model") or DEFAULT_MODEL
        self.sandbox      : bool = bool(cfg.get("sandbox", False))
        self.max_length   : int  = int(cfg.get("max_length", 1500))
        self.system_prompt: str  = cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

        # scope_key("{app_id}:{openid/group_openid}") → session_id
        self._sessions: Dict[str, str] = {}

        # scope_key → 当前正在运行的生成 Task（用于 /stop）
        self._tasks: Dict[str, asyncio.Task] = {}

        self._client = QQBotClient(self.app_id, self.app_secret, self.sandbox)
        self._client.on_message(self._on_message)

    # ── Session 管理 ─────────────────────────────────────────────────────

    def _scope_key(self, scope_id: str) -> str:
        """全局唯一键，防止多账号的同一 openid 互相碰撞。"""
        return f"{self.app_id}:{scope_id}"

    def _get_session(self, scope_id: str, label: str) -> str:
        key = self._scope_key(scope_id)
        if key not in self._sessions:
            s = _shared_manager.create_session(
                name=f"[{self.name}] {label}_{scope_id[:12]}",
                model=self.model,
                system_prompt=self.system_prompt,
            )
            self._sessions[key] = s.id
            log.info("[%s] 新建 session  %s → %s", self.name, scope_id[:12], s.id[:8])
        return self._sessions[key]

    def _reset_session(self, scope_id: str, label: str) -> str:
        """/new 指令：丢弃旧 session，强制新建。"""
        key = self._scope_key(scope_id)
        old_sid = self._sessions.pop(key, None)
        if old_sid:
            log.info("[%s] /new 丢弃旧 session %s", self.name, old_sid[:8])
        return self._get_session(scope_id, label)

    # ── 指令识别 ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_cmd(text: str) -> Optional[str]:
        """识别 /stop、/new 等内置指令，返回指令名；普通消息返回 None。"""
        t = text.strip().lower()
        if t in ("/stop", "/停止", "stop", "停止"):
            return "stop"
        if t in ("/new", "/新对话", "new", "新对话"):
            return "new"
        return None

    # ── AI 生成（同步，在线程池运行）────────────────────────────────────

    def _sync_generate(
        self, session_id: str, user_text: str, stop_event: threading.Event
    ) -> str:
        session = _shared_manager.get_session(session_id)
        if not session:
            return "抱歉，会话异常，请稍后重试。"

        full_text, pending = "", []

        # 第一轮：用户消息 → 模型
        for ev in _shared_manager.stream_chat(session=session, user_message=user_text):
            if stop_event.is_set():
                log.info("[%s] 生成被 /stop 中断（第一轮）", self.name)
                break
            if ev["type"] == "text":
                full_text += ev["content"]
            elif ev["type"] == "tool_call":
                pending = ev["tool_calls"]

        # 工具调用循环（最多 5 轮）
        for _r in range(5):
            if not pending or stop_event.is_set():
                break
            names = [tc.get("function", {}).get("name") for tc in pending]
            log.info("[%s] 执行工具 round=%d  %s", self.name, _r + 1, names)
            for tc in pending:
                if stop_event.is_set():
                    break
                fn  = tc.get("function", {})
                out = _shared_manager._executor.execute(
                    fn.get("name", ""), fn.get("arguments", {}))
                _shared_manager._store.append_message(
                    session, {"role": "tool", "name": fn.get("name", ""), "content": out})
            if stop_event.is_set():
                break
            pending = []
            for ev in _shared_manager._stream_internal(
                session, _shared_manager._build_model_messages(session)
            ):
                if stop_event.is_set():
                    break
                if ev["type"] == "text":
                    full_text += ev["content"]
                elif ev["type"] == "tool_call":
                    pending = ev["tool_calls"]

        text = full_text.strip() or "（AI 无回复）"
        if len(text) > self.max_length:
            text = text[:self.max_length] + "\n…（内容过长已截断）"
        if stop_event.is_set() and full_text:
            text += "\n[已中断]"
        return text

    # ── AI 生成（异步包装）───────────────────────────────────────────────

    async def _generate(self, scope_key: str, session_id: str, user_text: str) -> str:
        stop_event = threading.Event()
        loop = asyncio.get_event_loop()

        async def _run() -> str:
            try:
                return await loop.run_in_executor(
                    None, self._sync_generate, session_id, user_text, stop_event
                )
            except asyncio.CancelledError:
                stop_event.set()   # 通知线程尽快退出生成循环
                raise

        task = asyncio.create_task(_run())
        self._tasks[scope_key] = task
        try:
            return await task
        except asyncio.CancelledError:
            return "（已中断）"
        finally:
            self._tasks.pop(scope_key, None)

    # ── 中断辅助 ─────────────────────────────────────────────────────────

    def _cancel_task(self, scope_key: str) -> bool:
        """取消 scope_key 对应的生成 Task，返回是否存在正在运行的任务。"""
        task = self._tasks.get(scope_key)
        if task and not task.done():
            task.cancel()
            return True
        return False

    # ── 消息分发 ─────────────────────────────────────────────────────────

    async def _on_message(self, event: Dict[str, Any]) -> None:
        etype = event.get("_event_type", "")
        if etype == "C2C_MESSAGE_CREATE":
            await self._handle_c2c(event)
        elif etype == "GROUP_AT_MESSAGE_CREATE":
            await self._handle_group(event)

    # ── 私聊处理 ─────────────────────────────────────────────────────────

    async def _handle_c2c(self, event: Dict[str, Any]) -> None:
        author  = event.get("author", {})
        openid  = author.get("user_openid") or author.get("id", "")
        msg_id  = event.get("id", "")
        content = (event.get("content") or "").strip()
        if not content or not openid:
            return

        scope_key = self._scope_key(openid)
        cmd = self._parse_cmd(content)

        if cmd == "stop":
            if self._cancel_task(scope_key):
                log.info("[%s] /stop 私聊  user=%s", self.name, openid[:12])
                await self._client.send_c2c(openid, "⏹ 已中断生成。", msg_id)
            else:
                await self._client.send_c2c(openid, "当前没有正在生成的内容。", msg_id)
            return

        if cmd == "new":
            self._cancel_task(scope_key)       # 有生成中的先中断
            self._reset_session(openid, "c2c")
            await self._client.send_c2c(openid, "✅ 已开启新对话，之前的记忆已清除。", msg_id)
            return

        log.info("[%s] 私聊  user=%s  text=%.50s", self.name, openid[:12], content)
        sid   = self._get_session(openid, "c2c")
        reply = await self._generate(scope_key, sid, content)
        await self._client.send_c2c(openid, reply, msg_id)

    # ── 群聊处理 ─────────────────────────────────────────────────────────

    async def _handle_group(self, event: Dict[str, Any]) -> None:
        group_openid = event.get("group_openid", "")
        msg_id       = event.get("id", "")
        content      = (event.get("content") or "").lstrip()
        if not content or not group_openid:
            return

        scope_key = self._scope_key(group_openid)
        cmd = self._parse_cmd(content)

        if cmd == "stop":
            if self._cancel_task(scope_key):
                log.info("[%s] /stop 群聊  group=%s", self.name, group_openid[:12])
                await self._client.send_group(group_openid, "⏹ 已中断生成。", msg_id)
            else:
                await self._client.send_group(group_openid, "当前没有正在生成的内容。", msg_id)
            return

        if cmd == "new":
            self._cancel_task(scope_key)
            self._reset_session(group_openid, "group")
            await self._client.send_group(group_openid, "✅ 已开启新对话，之前的记忆已清除。", msg_id)
            return

        log.info("[%s] 群消息  group=%s  text=%.50s", self.name, group_openid[:12], content)
        sid   = self._get_session(group_openid, "group")
        reply = await self._generate(scope_key, sid, content)
        await self._client.send_group(group_openid, reply, msg_id)

    # ── 启动 ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        log.info("[%s] 启动  AppID=%s  model=%s  sandbox=%s",
                 self.name, self.app_id, self.model, self.sandbox)
        await self._client.run()


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_bots_config(path: str) -> List[Dict[str, Any]]:
    """从 JSON 文件加载多 Bot 配置；找不到文件时从环境变量兜底（单账号）。"""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path} 必须是非空 JSON 数组")
        log.info("从 %s 加载了 %d 个 Bot 配置", path, len(data))
        return data

    app_id     = os.getenv("QQ_APP_ID", "").strip()
    app_secret = os.getenv("QQ_APP_SECRET", "").strip()
    if app_id and app_secret:
        log.info("未找到 %s，从环境变量加载单账号配置", path)
        return [{
            "app_id":     app_id,
            "app_secret": app_secret,
            "sandbox":    os.getenv("QQ_SANDBOX", "false").lower() == "true",
            "model":      os.getenv("QQ_DEFAULT_MODEL", DEFAULT_MODEL),
            "max_length": int(os.getenv("QQ_MAX_LENGTH", "1500")),
        }]

    raise FileNotFoundError(
        f"找不到配置文件 {path}，也未设置 QQ_APP_ID / QQ_APP_SECRET。\n"
        f"请创建 bots.json，或在 .env 中配置单账号。"
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

async def main(config_path: str) -> None:
    configs   = load_bots_config(config_path)
    instances = [BotInstance(cfg) for cfg in configs]

    log.info("=" * 55)
    log.info("共启动 %d 个 QQ Bot", len(instances))
    for b in instances:
        log.info("  · [%s]  AppID=%s  model=%s", b.name, b.app_id, b.model)
    log.info("支持指令: /stop（中断生成）  /new（新对话）")
    log.info("=" * 55)

    await asyncio.gather(*[inst.run() for inst in instances], return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多账号 QQ Bot")
    parser.add_argument("--config", default="bots.json", help="Bot 配置文件路径")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.config))
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[启动失败] {e}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("所有 QQ Bot 已停止")