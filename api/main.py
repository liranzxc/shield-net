import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from api.shield_net.decorators import shield_net


# Define request body model
class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[str]] = []  # History of previous messages

# Define response body model
class ChatResponse(BaseModel):
    response: str
    updated_history: List[str]  # Updated history including the new response

# Initialize FastAPI app
app = FastAPI()


@shield_net
def invoke(prompt: str, history: List[str]) -> (str, List[str]):
    """
    Helper function to process the prompt and generate a response,
    while updating the history.
    """
    # Generate a simple response (placeholder for actual logic)
    response = f"You said: {prompt}"
    # Update the history
    updated_history = history + [prompt, response]
    return response, updated_history

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Responds to a chat prompt and maintains a history of messages.
    """
    prompt = request.prompt
    history = request.history

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Use the helper function to process the prompt and update the history
    response, updated_history = invoke(prompt, history)
    return ChatResponse(response=response, updated_history=updated_history)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
