import datetime
import json
import time
from fastapi import FastAPI
from openai import AsyncOpenAI
import os
from starlette.responses import StreamingResponse
from app.types.vapi import ChatRequest
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


@app.post("/chat/completions")
async def chat_completion_stream(vapi_payload: ChatRequest):
    """
    Endpoint to handle chat completions streaming from OpenAI's API.
    
    Args:
        request (ChatRequest): The request body containing model, messages, temperature, and tools.

    Returns:
        StreamingResponse: A streaming response with the chat completion data.
    """
    try:
        response = await client.chat.completions.create(
            model=vapi_payload.model,
            messages=vapi_payload.messages,
            temperature=vapi_payload.temperature,
            tools=vapi_payload.tools,
            stream=True,
        )

        async def event_stream():
            try:
                async for chunk in response:
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Error during response streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        return StreamingResponse(
            f"data: {json.dumps({'error': str(e)})}\n\n", media_type="text/event-stream"
        )