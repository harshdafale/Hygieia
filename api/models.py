"""API request schemas."""
from pydantic import BaseModel, Field

class SessionIn(BaseModel):
    crew_id: int = Field(ge=1)

class AskIn(BaseModel):
    session_id: int = Field(ge=1)
    question: str = Field(min_length=2, max_length=2000)

class CopilotIn(BaseModel):
    session_id: int = Field(ge=1)
    question: str = Field(min_length=2, max_length=2000)

class DiagnoseIn(BaseModel):
    session_id: int = Field(ge=1)
    symptoms: str = Field(min_length=2, max_length=2000)
    severity: int = Field(ge=1, le=10)
    duration: str = Field(max_length=80)
    worsening: bool = False
    vitals: dict[str, str | float | int] = {}

class FeedbackIn(BaseModel):
    message_id: int = Field(ge=1)
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=1000)
