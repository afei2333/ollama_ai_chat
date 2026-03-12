"""
qqbot_client.py — 腾讯 QQ 机器人官方 API 客户端

直接对接腾讯开放平台，无需任何第三方框架。
实现了完整的 WebSocket 长连接生命周期管理：
  - access_token 自动刷新（7200秒有效期，提前60秒续期）
  - WebSocket 鉴权、心跳维持、断线自动重连
  - 私聊（C2C）+ QQ群（GROUP）消息收发
  - Session Resume（断线续连，补发遗漏事件）

对外只暴露两个接口：
  QQBotClient(app_id, app_secret)
  client.on_message(handler)   # 注册消息处理回调
  await client.run()           # 启动，永久运行

发送消息：
  await client.send_c2c(openid, content, msg_id)    # 回复私聊
  await client.send_group(group_openid, content, msg_id, member_openid)  # 回复群消息
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Coroutine, Dict, Optional

import aiohttp

from logger import get_logger

log = get_logger(__name__)

# ── 腾讯官方 API 地址 ──────────────────────────────────────────────────────
_TOKEN_URL    = "https://bots.qq.com/app/getAppAccessToken"
_GATEWAY_URL  = "https://api.sgroup.qq.com/gateway"
_API_BASE     = "https://api.sgroup.qq.com"

# WebSocket OpCode
_OP_DISPATCH        = 0   # 服务端推送事件
_OP_HEARTBEAT       = 1   # 客户端发心跳
_OP_IDENTIFY        = 2   # 客户端鉴权
_OP_RESUME          = 6   # 客户端恢复连接
_OP_RECONNECT       = 7   # 服务端要求重连
_OP_INVALID_SESSION = 9   # 无效 Session
_OP_HELLO           = 10  # 服务端发送心跳间隔
_OP_HEARTBEAT_ACK   = 11  # 服务端确认心跳

# 订阅事件类型（GROUP_AND_C2C_EVENT = 1<<25）
_INTENT_GROUP_AND_C2C = 1 << 25

MessageHandler = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


class QQBotClient:
    """
    腾讯 QQ 官方机器人客户端。
    维护 access_token + WebSocket 长连接，自动处理心跳/重连/续连。
    """

    def __init__(self, app_id: str, app_secret: str, sandbox: bool = False) -> None:
        self.app_id     = app_id
        self.app_secret = app_secret
        self.sandbox    = sandbox

        self._token: str           = ""
        self._token_expire: float  = 0.0   # unix timestamp，token 过期时刻
        self._session_id: str      = ""
        self._last_seq: int        = 0
        self._ws_url: str          = ""

        self._http: Optional[aiohttp.ClientSession] = None
        self._ws:   Optional[aiohttp.ClientWebSocketResponse] = None
        self._hb_task: Optional[asyncio.Task] = None

        self._handlers: list[MessageHandler] = []

    # -----------------------------------------------------------------------
    # 公开接口
    # -----------------------------------------------------------------------

    def on_message(self, handler: MessageHandler) -> MessageHandler:
        """注册消息事件回调（可注册多个）。handler 接收原始事件 dict。"""
        self._handlers.append(handler)
        return handler

    async def send_c2c(
        self,
        openid: str,
        content: str,
        msg_id: str,
        msg_seq: int = 1,
    ) -> Dict[str, Any]:
        """
        回复用户私聊（C2C）消息。
        必须在收到用户消息后的 5 分钟内调用（被动回复），
        否则需要申请主动消息权限。
        """
        url  = f"{_API_BASE}/v2/users/{openid}/messages"
        body = {
            "content":  content,
            "msg_type": 0,          # 0=文本
            "msg_id":   msg_id,
            "msg_seq":  msg_seq,
        }
        return await self._post(url, body)

    async def send_group(
        self,
        group_openid: str,
        content: str,
        msg_id: str,
        msg_seq: int = 1,
    ) -> Dict[str, Any]:
        """
        回复群消息（被动回复）。
        必须在收到群消息后的 5 分钟内调用。
        """
        url  = f"{_API_BASE}/v2/groups/{group_openid}/messages"
        body = {
            "content":  content,
            "msg_type": 0,
            "msg_id":   msg_id,
            "msg_seq":  msg_seq,
        }
        return await self._post(url, body)

    async def run(self) -> None:
        """启动客户端，永久运行，自动重连。"""
        async with aiohttp.ClientSession() as session:
            self._http = session
            while True:
                try:
                    await self._connect_and_listen()
                except asyncio.CancelledError:
                    log.info("QQBotClient 收到取消信号，停止运行")
                    return
                except Exception as exc:
                    log.error("连接异常，5 秒后重试: %s", exc, exc_info=True)
                    await asyncio.sleep(5)

    # -----------------------------------------------------------------------
    # access_token 管理
    # -----------------------------------------------------------------------

    async def _ensure_token(self) -> str:
        """确保 access_token 有效（不足 60 秒过期时自动刷新）。"""
        if time.time() < self._token_expire - 60:
            return self._token

        log.info("正在获取 access_token …")
        async with self._http.post(
            _TOKEN_URL,
            json={"appId": self.app_id, "clientSecret": self.app_secret},
        ) as resp:
            data = await resp.json()

        token      = data.get("access_token", "")
        expires_in = int(data.get("expires_in", 7200))

        if not token:
            raise RuntimeError(f"获取 access_token 失败: {data}")

        self._token        = token
        self._token_expire = time.time() + expires_in
        log.info("access_token 已刷新，有效期 %d 秒", expires_in)
        return self._token

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization":  f"QQBot {self._token}",
            "X-Union-Appid":  self.app_id,
            "Content-Type":   "application/json",
        }

    # -----------------------------------------------------------------------
    # HTTP 工具
    # -----------------------------------------------------------------------

    async def _post(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_token()
        async with self._http.post(url, json=body, headers=self._auth_headers()) as resp:
            data = await resp.json()
            if resp.status not in (200, 202):
                log.warning("POST %s 返回 %d: %s", url, resp.status, data)
            return data

    async def _get(self, url: str) -> Dict[str, Any]:
        await self._ensure_token()
        async with self._http.get(url, headers=self._auth_headers()) as resp:
            return await resp.json()

    # -----------------------------------------------------------------------
    # WebSocket 生命周期
    # -----------------------------------------------------------------------

    async def _get_gateway(self) -> str:
        """获取 WSS 网关地址。"""
        if self.sandbox:
            return "wss://sandbox.api.sgroup.qq.com/websocket"
        data = await self._get(_GATEWAY_URL)
        url  = data.get("url", "")
        if not url:
            raise RuntimeError(f"获取 Gateway 失败: {data}")
        log.info("Gateway: %s", url)
        return url

    async def _connect_and_listen(self) -> None:
        """建立 WebSocket 连接并进入事件监听循环。"""
        await self._ensure_token()
        ws_url = await self._get_gateway()

        async with self._http.ws_connect(ws_url) as ws:
            self._ws = ws
            log.info("WebSocket 已连接: %s", ws_url)

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_ws_message(json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    log.error("WebSocket 错误: %s", msg.data)
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    log.warning("WebSocket 连接关闭")
                    break

            if self._hb_task:
                self._hb_task.cancel()
                self._hb_task = None

    async def _handle_ws_message(self, payload: Dict[str, Any]) -> None:
        """分发 WebSocket 消息到对应处理函数。"""
        op = payload.get("op")
        s  = payload.get("s")   # 事件序号

        if s is not None:
            self._last_seq = s

        if op == _OP_HELLO:
            interval = payload["d"]["heartbeat_interval"]
            log.info("收到 Hello，心跳间隔 %d ms", interval)
            self._hb_task = asyncio.create_task(
                self._heartbeat_loop(interval / 1000)
            )
            await self._identify_or_resume()

        elif op == _OP_DISPATCH:
            await self._on_dispatch(payload)

        elif op == _OP_RECONNECT:
            log.warning("服务端要求重连（OP7）")
            await self._ws.close()

        elif op == _OP_INVALID_SESSION:
            log.warning("Session 失效（OP9），清除 session_id 重新鉴权")
            self._session_id = ""
            await self._identify_or_resume()

        elif op == _OP_HEARTBEAT_ACK:
            log.debug("心跳 ACK")

    async def _identify_or_resume(self) -> None:
        """首次连接发 IDENTIFY；断线重连且有 session_id 时发 RESUME。"""
        if self._session_id:
            log.info("尝试 Resume session=%s seq=%d", self._session_id, self._last_seq)
            payload = {
                "op": _OP_RESUME,
                "d": {
                    "token":      f"QQBot {self._token}",
                    "session_id": self._session_id,
                    "seq":        self._last_seq,
                },
            }
        else:
            log.info("发送 Identify（首次鉴权）")
            payload = {
                "op": _OP_IDENTIFY,
                "d": {
                    "token":   f"QQBot {self._token}",
                    "intents": _INTENT_GROUP_AND_C2C,
                    "shard":   [0, 1],
                    "properties": {},
                },
            }
        await self._ws.send_str(json.dumps(payload))

    async def _heartbeat_loop(self, interval: float) -> None:
        """按照服务端指定的间隔周期发送心跳。"""
        try:
            while True:
                await asyncio.sleep(interval)
                await self._ws.send_str(json.dumps({"op": _OP_HEARTBEAT, "d": self._last_seq}))
                log.debug("心跳已发送 seq=%d", self._last_seq)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("心跳发送失败: %s", exc)

    # -----------------------------------------------------------------------
    # 事件派发
    # -----------------------------------------------------------------------

    async def _on_dispatch(self, payload: Dict[str, Any]) -> None:
        """处理 OP=0 的事件推送。"""
        t    = payload.get("t", "")
        data = payload.get("d", {})

        if t == "READY":
            self._session_id = data.get("session_id", "")
            user = data.get("user", {})
            log.info(
                "机器人就绪 ✓  session=%s  bot=%s(%s)",
                self._session_id, user.get("username"), user.get("id"),
            )
            return

        if t == "RESUMED":
            log.info("Session 恢复成功，开始补发事件")
            return

        # 只处理私聊和群聊消息
        if t in ("C2C_MESSAGE_CREATE", "GROUP_AT_MESSAGE_CREATE"):
            log.info("收到事件 t=%s", t)
            # 为 handler 注入事件类型，方便区分
            data["_event_type"] = t
            for handler in self._handlers:
                asyncio.create_task(self._safe_call(handler, data))
        else:
            log.debug("忽略事件 t=%s", t)

    @staticmethod
    async def _safe_call(handler: MessageHandler, data: Dict[str, Any]) -> None:
        """安全调用 handler，捕获异常避免单个处理失败影响后续。"""
        try:
            await handler(data)
        except Exception as exc:
            log.error("消息处理器异常: %s", exc, exc_info=True)