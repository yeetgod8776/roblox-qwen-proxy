from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os
import time

app = FastAPI()

# Get your Scitely API key from environment variables
SCITELY_API_KEY = os.getenv("SCITELY_API_KEY")
if not SCITELY_API_KEY:
    raise RuntimeError("SCITELY_API_KEY environment variable is required")

SCITELY_URL = "https://api.scitely.com/v1/chat/completions"

@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Extract parameters (with defaults)
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 1024)
    model = body.get("model", "qwen3-max")  # ignored, we always use qwen3-max

    # Validate messages
    if not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' field")

    headers = {
        "Authorization": f"Bearer {SCITELY_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            scitely_resp = await client.post(
                SCITELY_URL,
                json={
                    "model": "qwen3-max",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            scitely_resp.raise_for_status()
            data = scitely_resp.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("error", {}).get("message", str(e))
            raise HTTPException(status_code=e.response.status_code, detail=f"Scitely API error: {error_detail}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

    # Convert Scitely response → OpenAI format
    choice = data["choices"][0]
    openai_response = {
        "id": data.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen3-max",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": choice["message"]["content"]
            },
            "finish_reason": choice.get("finish_reason", "stop")
        }],
        "usage": {
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": data.get("usage", {}).get("total_tokens", 0)
        }
    }

    return JSONResponse(content=openai_response)
