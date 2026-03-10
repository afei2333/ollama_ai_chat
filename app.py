"""
app.py — FastAPI 入口

变更说明：
  - 新增全局异常处理器（Exception → 500 JSON），统一错误响应结构
  - _get_session_or_404 改为工厂函数，利用路径参数自动注入 session_id，
    消除每个路由手动传 session_id 的样板代码
  - /sessions/{session_id}/messages 使用 Depends 复用 session 获取逻辑
  - SSE_HEADERS 补充 X-Accel-Buffering: no，避免 Nginx 代理时缓冲 SSE 流
  - lifespan / 路由分组保持不变
"""

import json
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ollama_client import ChatManager, SessionState, TAVILY_API_KEY
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# 应用初始化
# ---------------------------------------------------------------------------

manager = ChatManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Local AI Assistant 启动 ===")
    log.info("Tavily 联网搜索: %s", "已启用" if TAVILY_API_KEY else "未配置（缺少 TAVILY_API_KEY）")
    yield
    log.info("=== 服务关闭，卸载模型 ===")
    manager.unload_model()


app = FastAPI(title="Local AI Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# 全局异常处理
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("未捕获异常 %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"},
    )


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    max_turns: int = Field(default=8, ge=1, le=50)


class UpdateSessionRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    max_turns: Optional[int] = Field(default=None, ge=1, le=50)


class StreamChatRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=8)
    use_rag: bool = True


class ExecuteToolRequest(BaseModel):
    tool_calls: List[dict]


class AddDocRequest(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# 公共辅助
# ---------------------------------------------------------------------------

def _session_summary(session: SessionState) -> dict:
    return {
        "id": session.id,
        "name": session.name,
        "model": session.model,
        "max_turns": session.max_turns,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": len(session.messages),
    }


def get_session_dep(session_id: str) -> SessionState:
    """
    路径参数工厂依赖：FastAPI 会自动将路由中的 {session_id} 注入此函数。
    不存在时抛 404，避免在每个路由函数中重复判断。
    """
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return session


def _sse_generator(generator, session_id: str):
    """将内部事件流包装成 SSE 格式字节流。"""
    try:
        for item in generator:
            if item["type"] == "text":
                payload = json.dumps({"text": item["content"]}, ensure_ascii=False)
                yield f"event: token\ndata: {payload}\n\n"
            elif item["type"] == "tool_call":
                payload = json.dumps({"tool_calls": item["tool_calls"]}, ensure_ascii=False)
                yield f"event: tool_call\ndata: {payload}\n\n"

        done_payload = json.dumps({"session_id": session_id}, ensure_ascii=False)
        yield f"event: done\ndata: {done_payload}\n\n"

    except Exception as err:
        log.error("SSE 流异常 session=%s: %s", session_id[:8], err, exc_info=True)
        error_payload = json.dumps({"error": str(err)}, ensure_ascii=False)
        yield f"event: error\ndata: {error_payload}\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",   # 禁止 Nginx 缓冲 SSE
}


def _streaming_response(generator, session_id: str) -> StreamingResponse:
    return StreamingResponse(
        _sse_generator(generator, session_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# 路由：静态页面
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# 路由：模型
# ---------------------------------------------------------------------------

@app.get("/models")
def list_models():
    return {"models": manager.list_models()}


@app.get("/config/tools")
def get_tool_config():
    """返回各工具的可用状态，供前端展示提示。"""
    return {
        "tools": {
            "run_shell": {"enabled": True, "description": "本地终端命令"},
            "web_search": {
                "enabled": bool(TAVILY_API_KEY),
                "description": "网络搜索 (Tavily)",
            },
        }
    }


# ---------------------------------------------------------------------------
# 路由：会话管理
# ---------------------------------------------------------------------------

@app.get("/sessions")
def list_sessions():
    return {"sessions": [_session_summary(s) for s in manager.list_sessions()]}


@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    session = manager.create_session(name=req.name, model=req.model, max_turns=req.max_turns)
    return {"session": _session_summary(session)}


@app.patch("/sessions/{session_id}")
def update_session(session_id: str, req: UpdateSessionRequest):
    session = manager.update_session(
        session_id=session_id,
        model=req.model,
        max_turns=req.max_turns,
        name=req.name,
    )
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session": _session_summary(session)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"deleted": True}


@app.get("/sessions/{session_id}/messages")
def session_messages(session: SessionState = Depends(get_session_dep)):
    return {"messages": session.messages}


# ---------------------------------------------------------------------------
# 路由：流式对话
# ---------------------------------------------------------------------------

@app.post("/sessions/{session_id}/stream")
def stream_chat(
    req: StreamChatRequest,
    session: SessionState = Depends(get_session_dep),
):
    gen = manager.stream_chat(
        session=session,
        user_message=req.message,
        top_k=req.top_k,
        use_rag=req.use_rag,
    )
    return _streaming_response(gen, session.id)


@app.post("/sessions/{session_id}/tool_stream")
def stream_tool_chat(
    req: ExecuteToolRequest,
    session: SessionState = Depends(get_session_dep),
):
    """用户确认工具执行后调用此接口，将工具输出喂回模型继续生成。"""
    gen = manager.execute_and_stream(session, req.tool_calls)
    return _streaming_response(gen, session.id)


# ---------------------------------------------------------------------------
# 路由：知识库
# ---------------------------------------------------------------------------

@app.get("/kb/docs")
def list_kb_docs():
    docs = manager.kb.list_documents()
    return {
        "documents": [
            {"id": d.id, "title": d.title, "content": d.content, "created_at": d.created_at}
            for d in docs
        ]
    }


@app.post("/kb/docs")
def add_kb_doc(req: AddDocRequest):
    doc = manager.kb.add_document(req.title, req.content)
    return {"document": {"id": doc.id, "title": doc.title, "content": doc.content, "created_at": doc.created_at}}


@app.delete("/kb/docs/{doc_id}")
def delete_kb_doc(doc_id: str):
    if not manager.kb.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="document not found")
    return {"deleted": True}
