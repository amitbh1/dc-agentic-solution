from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SESSION = r"^[A-Za-z0-9_-]{1,64}$"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, pattern=SESSION)


class ApproveRequest(BaseModel):
    session_id: str | None = Field(default=None, pattern=SESSION)
    approve: bool


class FeedbackRequest(BaseModel):
    session_id: str | None = Field(default=None, pattern=SESSION)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)
