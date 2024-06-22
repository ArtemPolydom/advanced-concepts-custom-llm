from typing import Optional, List
from pydantic import BaseModel


class SystemMessage(BaseModel):
    role: str
    content: str


class AssistantMessage(BaseModel):
    role: str
    content: str


class UserMessage(BaseModel):
    role: str
    content: str


class CallDetails(BaseModel):
    id: str
    orgId: str
    createdAt: str
    updatedAt: str
    type: str
    status: str
    assistantId: str
    webCallUrl: str


class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    temperature: float
    stream: bool
    tools: List[dict]
    max_tokens: int
    call: CallDetails
    metadata: Optional[dict] = None
