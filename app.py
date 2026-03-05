import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ollama_client import ChatManager, SessionState

app = FastAPI(title="Local AI Assistant")
manager = ChatManager()

app.mount("/static", StaticFiles(directory="static"), name="static")


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


class AddDocRequest(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


def _session_summary(session: SessionState):
    return {
        "id": session.id,
        "name": session.name,
        "model": session.model,
        "max_turns": session.max_turns,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": len(session.messages),
    }


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/models")
def list_models():
    return {"models": manager.list_models()}


@app.get("/sessions")
def list_sessions():
    sessions = [_session_summary(s) for s in manager.list_sessions()]
    return {"sessions": sessions}


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


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str):
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return {"messages": session.messages}


@app.post("/sessions/{session_id}/stream")
def stream_chat(session_id: str, req: StreamChatRequest):
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    def event_stream():
        try:
            for text in manager.stream_chat(
                session=session,
                user_message=req.message,
                top_k=req.top_k,
                use_rag=req.use_rag,
            ):
                yield f"event: token\ndata: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            yield (
                "event: done\n"
                f"data: {json.dumps({'session_id': session.id}, ensure_ascii=False)}\n\n"
            )
        except Exception as err:
            yield f"event: error\ndata: {json.dumps({'error': str(err)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/kb/docs")
def list_kb_docs():
    docs = manager.kb.list_documents()
    return {
        "documents": [
            {
                "id": d.id,
                "title": d.title,
                "content": d.content,
                "created_at": d.created_at,
            }
            for d in docs
        ]
    }


@app.post("/kb/docs")
def add_kb_doc(req: AddDocRequest):
    doc = manager.kb.add_document(req.title, req.content)
    return {
        "document": {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "created_at": doc.created_at,
        }
    }


@app.delete("/kb/docs/{doc_id}")
def delete_kb_doc(doc_id: str):
    ok = manager.kb.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="document not found")
    return {"deleted": True}