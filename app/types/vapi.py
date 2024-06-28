from __future__ import annotations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class Parameters(BaseModel):
    type: str
    properties: Dict[str, Any]


class Function(BaseModel):
    name: str
    description: str
    parameters: Optional[Parameters] = None


class Tool(BaseModel):
    type: str
    function: Function


class Customer(BaseModel):
    number: str


class Call(BaseModel):
    id: str
    orgId: str
    createdAt: str
    updatedAt: str
    type: str
    status: str
    assistantId: str
    customer: Optional[Customer] = None
    phoneNumberId: Optional[str] = None
    phoneCallProvider: Optional[str] = None
    phoneCallProviderId: Optional[str] = None
    phoneCallTransport: Optional[str] = None


class PhoneNumber(BaseModel):
    id: str
    orgId: str
    number: str
    createdAt: str
    updatedAt: str
    twilioAccountSid: str
    twilioAuthToken: str
    name: str
    provider: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float
    tools: List[Tool]
    stream: bool
    max_tokens: int
    call: Call
    phoneNumber: Optional[PhoneNumber] = None
    customer: Optional[Customer] = None
    metadata: Dict[str, Any]
