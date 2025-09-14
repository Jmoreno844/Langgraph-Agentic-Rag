from typing import List, Literal
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Role of the message author"
    )
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Session identifier to correlate conversation state",
        examples=["session-123"],
    )
    messages: List[ChatMessage] = Field(
        ..., description="Ordered chat history; last item is the latest user message"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "session-123",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hi!"},
                        {
                            "role": "user",
                            "content": "What are the available product categories?",
                        },
                    ],
                }
            ]
        }
    }


def convert_chat_messages(messages: List[ChatMessage]):
    converted = []
    for m in messages:
        if m.role == "user":
            converted.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            converted.append(AIMessage(content=m.content))
        else:
            converted.append(SystemMessage(content=m.content))
    return converted


router = APIRouter()


@router.post(
    "/chat",
    summary="Stream chat responses (SSE)",
    tags=["chat"],
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events stream with incremental model output.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        503: {"description": "Chat runtime not initialized"},
    },
)
async def stream_chat(request: Request, body: ChatRequest):
    """Streams model responses over Server-Sent Events (SSE).

    Notes:
    - This endpoint streams events with media type `text/event-stream`.
    - The server uses the provided `session_id` to maintain conversation state.
    - Only the latest message may be forwarded to the runtime when appropriate.
    """
    if (
        not hasattr(request.app.state, "graph_app")
        or request.app.state.graph_app is None
    ):
        raise HTTPException(status_code=503, detail="Chat runtime not initialized")

    # Prepare payload for the graph runtime
    payload = {
        "messages": convert_chat_messages(body.messages[-1:]),
        "has_been_rewritten": False,
    }
    config = {"configurable": {"thread_id": body.session_id}}

    async def event_generator():
        # Initial comment to establish SSE
        yield "event: start\n\n"
        async for chunk in request.app.state.graph_app.astream(payload, config=config):
            try:
                data = json.dumps(chunk, default=str)
            except Exception:
                data = str(chunk)
            yield f"data: {data}\n\n"
        # Explicit end event
        yield "event: end\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
