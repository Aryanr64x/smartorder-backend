from pydantic import BaseModel
from typing import List, Literal


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    query: str
    # Prior conversation turns sent by the client (oldest first). The backend is
    # stateless, so the frontend owns the transcript and replays it each request.
    history: List[ChatTurn] = []