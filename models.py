"""
models.py — 数据模型定义（纯数据类，无业务逻辑）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


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
