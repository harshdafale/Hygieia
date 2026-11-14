"""API request schemas."""
from pydantic import BaseModel, Field

class SessionIn(BaseModel):
    crew_id: int = Field(ge=1)

class AskIn(BaseModel):
    session_id: int = Field(ge=1)
    question: str = Field(min_length=2, max_length=2000)

class FeedbackIn(BaseModel):
    message_id: int = Field(ge=1)
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=1000)
