import asyncio
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import os

from starlette.requests import Request

from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


async def get_payment_link():
    await asyncio.sleep(10)
    print("Getting payment link")
    return f"https://example.com/pay/333333"


tools = {'get_payment_link': get_payment_link}


class ChatCompletionRequestMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatCompletionRequestMessage]
    temperature: Optional[float] = Field(default=1.0)
    tools: Optional[list[str]] = None


client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


class ToolSchema(BaseModel):
    name: str
    description: str
    # parameters: dict


class ToolModel(BaseModel):
    type: str = "function"
    function: ToolSchema


tool_functions = {
    "get_payment_link": get_payment_link
}
empty_chunk = 'data: {"id": "chatcmpl-ArsKmBkHCXXPwinpTcOsrJ5ul6fKw", "choices": [{"delta": {"content": "<flash>Wait a minute please</flash>", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1737403608, "model": "gpt-3.5-turbo-0125", "object": "chat.completion.chunk", "service_tier": "default", "system_fingerprint": null, "usage": null}'


@app.post("/chat/completions")
async def get_openai_response(request: Request):
    try:
        # Extract the JSON body from the incoming request
        body = await request.json()
        print(f"Received Request Body: {json.dumps(body, indent=4)}")

        # Parse the relevant parts of the request body
        model = body.get('model')
        messages = body.get('messages')
        temperature = body.get('temperature', 1.0)
        tools = body.get('tools', [])

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_payment_link",
                    "description": "get_payment_link",
                    "parameters": {}
                }
            }, ]
        tools.extend(openai_tools)

        # Transform messages to the expected format
        chat_messages = [ChatCompletionRequestMessage(**msg) for msg in messages]

        # Prepare the arguments for the OpenAI request
        request_args = {
            "model": model,
            "messages": [msg.dict() for msg in chat_messages],
            "temperature": temperature,
            "stream": True,
        }

        # Add tools to the request only if it's not empty
        if tools:
            request_args["tools"] = tools

        # Create the request to OpenAI client
        response = await client.chat.completions.create(**request_args)

        # Stream the response back to the client
        async def event_stream():
            arguments_str = ""
            tool_call_info = None
            async for chunk in response:
                chunk_dict = chunk.model_dump()
                print(f"Chunk: {chunk_dict}")

                choices = chunk_dict.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    tool_calls = delta.get("tool_calls", [])

                    if tool_calls:
                        for tool_call in tool_calls:
                            function = tool_call.get("function", {})
                            if function.get("name"):
                                tool_call_info = {
                                    "name": function["name"],
                                    "id": tool_call.get("id")
                                }
                            if function.get("arguments"):
                                arguments_str += function["arguments"]

                    finish_reason = choice.get("finish_reason")
                    if finish_reason == "tool_calls" and tool_call_info:
                        try:
                            arguments = json.loads(arguments_str)
                            function_name = tool_call_info["name"]
                            yield empty_chunk
                            if function_name in tool_functions:
                                if asyncio.iscoroutinefunction(tool_functions[function_name]):
                                    result = await tool_functions[function_name](**arguments)
                                else:
                                    result = tool_functions[function_name](**arguments)

                                function_message = {
                                    "role": "function",
                                    "name": function_name,
                                    "content": json.dumps(result)
                                }

                                follow_up_response = await client.chat.completions.create(
                                    model=request_args["model"],
                                    messages=request_args["messages"] + [function_message],
                                    temperature=request_args["temperature"],
                                    stream=True
                                )

                                n = 0

                                async for follow_up_chunk in follow_up_response:
                                    chunk = json.dumps(follow_up_chunk.model_dump())  # Dump to JSON string
                                    chunk_dict = json.loads(chunk)  # Convert back to dictionary for key access

                                    c = chunk_dict['choices'][0]['delta']['content']
                                    # if n == 0:
                                    #     chunk_dict['choices'][0]['delta']['content'] = '<flush>' + c
                                    #     n += 1
                                    # if c is None:
                                    #     chunk_dict['choices'][0]['delta']['content'] = '</flush>'
                                    d = f"data: {json.dumps(chunk_dict)}\n\n"
                                    print(d)
                                    yield d
                                # Reset for potential future tool calls
                                arguments_str = ""
                                tool_call_info = None
                            else:
                                # If the function is not in our list, stream the chunks directly
                                yield f"data: {json.dumps(chunk_dict)}\n\n"
                        except json.JSONDecodeError:
                            print(f"Failed to parse arguments: {arguments_str}")
                        except Exception as e:
                            print(f"Error executing function {function_name}: {e}")
                    else:
                        yield f"data: {json.dumps(chunk_dict)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8030, reload=True)
