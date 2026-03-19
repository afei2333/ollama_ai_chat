"""
models.py — 数据模型定义（纯数据类，无业务逻辑）

变更说明：
  - KnowledgeDoc 新增 to_dict / from_dict，统一 tokens 的序列化，
    消除 storage.py 中散落的手工转换代码。
  - SessionState 同理，from_dict 做向后兼容的字段默认值处理。
  - 两个模型均不依赖任何外部模块，保持"纯数据"定位。
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "tokens": sorted(self.tokens),  # list，JSON 友好
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeDoc":
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            tokens=set(data.get("tokens", [])),
            created_at=data["created_at"],
        )


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

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(
            id=data["id"],
            name=data["name"],
            model=data["model"],
            max_turns=data.get("max_turns", 8),
            system_prompt=data.get("system_prompt", ""),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=data.get("messages", []),
        )
