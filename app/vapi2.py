import json
from contextlib import asynccontextmanager
from typing import Callable

import httpx
from fastapi import FastAPI, HTTPException, APIRouter
from openai import AsyncOpenAI
import os

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
my_domain = os.environ["MY_BACKEND_URL"]


VAPI_API_KEY = os.environ.get("VAPI_API_KEY")
VAPI_ASSISTANT_ID = os.environ.get("VAPI_ASSISTANT_ID")

router = APIRouter()


@router.post("/chat/completions")
async def chat_completion_stream(request: Request):
    """Handle vapi custom llm chat completions"""
    try:
        body = await request.json()
        print(body)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=body.get('messages'),
            temperature=body.get('temperature'),
            tools=body.get('tools'),
            tool_choice=body.get('tool_choice'),
            stream=True,
        )
        complete_response = []

        async def event_stream():
            try:
                async for chunk in response:
                    chunk_dict = chunk.model_dump()
                    print(chunk)
                    if content := chunk_dict.get("choices", [{}])[0].get("delta", {}).get("content"):
                        complete_response.append(content)
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                if complete_response:
                    full_response = "".join(complete_response)
                    print(f"🤖 Complete streamed response:\n{full_response}")

                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Error during response streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        return StreamingResponse(
            f"data: {json.dumps({'error': str(e)})}\n\n", media_type="text/event-stream"
        )


# ----------------------------
OPENAI_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "check_weather",
        "description": "Check weather in any city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City for weather check"
                },
            },
            "required": ["city"]
        }
    }
}

# 1. Define base tool handler type
ToolHandler = Callable[[dict], str]

# 2. Create tool registry
TOOL_REGISTRY: dict[str, ToolHandler] = {}


# 5. Registration decorator
def register_tool(name: str, ):
    def decorator(func: ToolHandler):
        TOOL_REGISTRY[name] = func
        return func

    return decorator


# 6. Convert OpenAI tools to Vapi format
def openai_to_vapi_tool(openai_tool: dict, messages: list) -> dict:
    return {
        "type": "function",
        "messages": messages,
        "function": openai_tool["function"],
        "async": False,
        "server": {
            "url": f"{my_domain}/tools/{openai_tool['function']['name']}"
        }
    }


WEATHER_MESSAGES = [
    # {
    #     "type": "request-start",
    #     "content": "Checking weather, wait a minute"
    # },
    {
        "type": "request-failed",
        "content": "Failed to get weather"
    },
    {
        "type": "request-response-delayed",
        "content": "Wait a bit"
    }
]

VAPI_WEATHER_TOOL = openai_to_vapi_tool(
    OPENAI_WEATHER_TOOL,
    messages=WEATHER_MESSAGES
)


# 7. Register tool
@register_tool("check_weather")
def weather_handler(parameters: dict) -> str:
    city = parameters.get("city", "Unknown City")
    if city.lower() == "moscow":
        return f"Weather in {city}: 20°C, cloudy"
    # Replace with real API call
    return f"Weather in {city}: 25°C, sunny"


# request models
class FunctionCall(BaseModel):
    name: str
    arguments: dict


class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall


class VapiToolMessage(BaseModel):
    toolCalls: list[ToolCall]


class VapiToolRequest(BaseModel):
    message: VapiToolMessage


# 8. Handle tool requests from Vapi
@router.post("/tools/{tool_name}")
async def handle_tool(tool_name: str, request: VapiToolRequest):
    # Check for tool calls in the request
    if not request.message.toolCalls:
        raise HTTPException(status_code=400, detail="No tool calls provided")

    # Extract the first tool call (assuming single tool call per request)
    tool_call = request.message.toolCalls[0]
    tool_call_id = tool_call.id
    parameters = tool_call.function.arguments

    # Get the registered handler
    handler = TOOL_REGISTRY.get(tool_name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")

    try:
        # Execute the tool handler
        result = handler(parameters)
        return {
            "results": [{
                "toolCallId": tool_call_id,
                "result": result
            }]
        }
    except Exception as e:
        print(f"Error in {tool_name} handler: {e}")
        return {
            "results": [{
                "toolCallId": tool_call_id,
                "error": f"{tool_name} error: {str(e)}"
            }]
        }


# Update Vapi Assistant on Startup
async def update_vapi_assistant_tools():
    if not VAPI_API_KEY or not VAPI_ASSISTANT_ID:
        print("VAPI_API_KEY or VAPI_ASSISTANT_ID not set. Skipping tools update.")
        return

    vapi_tools = [VAPI_WEATHER_TOOL]  # Add other tools here if needed

    data = {
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-2",
            "language": "en",
        },
        "model": {
            "provider": "custom-llm",
            "url": f"{my_domain}",
            "model": "gpt-4o-mini",
            "tools": vapi_tools,
            "systemPrompt": "",
            "temperature": 0.6
        },
        "voice": {
            "provider": "11labs",
            "voiceId": "EXAVITQu4vr4xnSDxMaL",
        },
        "firstMessage": "Hi how can I help you?",

    }

    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {VAPI_API_KEY}",
            "Content-Type": "application/json"
        }
        url = f"https://api.vapi.ai/assistant/{VAPI_ASSISTANT_ID}"
        response = await client.patch(url, headers=headers, json=data)
        if response.is_error:
            print(f"Failed to update assistant tools: {response.status_code} {response.text}")
        else:
            print("Successfully updated Vapi assistant tools.")


@asynccontextmanager
async def lifespan(ap: FastAPI):
    # Startup logic
    print("Starting up...")
    await update_vapi_assistant_tools()  # Call your initialization function
    yield  # App runs here
    # Shutdown logic (add any cleanup if needed)
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.include_router(router)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("vapi2:app", host="localhost", port=8031, reload=True, log_level="debug")
