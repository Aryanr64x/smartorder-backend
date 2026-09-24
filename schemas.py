from pydantic import BaseModel
from typing import List, Literal, Optional


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    query: str
    # Prior conversation turns sent by the client (oldest first). The backend is
    # stateless, so the frontend owns the transcript and replays it each request.
    history: List[ChatTurn] = []
    # Both come from the QR code URL (`/chat?table=<id>&restaurant=<id>`).
    # table_id is authoritative — restaurant_id is a hint the backend validates
    # against the table before scoping retrieval. See tenancy.py.
    table_id: Optional[int] = None
    restaurant_id: Optional[int] = None