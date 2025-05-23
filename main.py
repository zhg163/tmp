import asyncio # Required for asyncio.gather in shutdown
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import uuid4

from config import OPENAI_API_KEY # For checking if API key is set
from db.clients import get_redis_client, get_mongo_client, create_redis_search_index
from db.memory_manager import MemoryManager
from langchain_utils.llm_chain import get_llm_chain
from redis.asyncio import Redis as AsyncRedis # For type hinting if needed, though clients.py uses redis.Redis

# --- Global Variables ---
# These will be initialized on startup
app_state: Dict[str, Any] = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str

class SearchRequest(BaseModel):
    query: str
    search_type: str = Field("semantic", pattern="^(semantic|full_text)$") # "semantic" or "full_text"
    session_id: Optional[str] = None # Optional: if you want to scope search to a session
    top_k: int = 5

class SearchResultItem(BaseModel):
    session_id: str
    turn_id: int
    role: str
    content: str
    timestamp: str # Keep as string for simplicity in response
    score: Optional[float] = None # Only for semantic search

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

# --- FastAPI App ---
app = FastAPI(title="AI Memory Chat API", version="0.1.0")

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Initialize resources on application startup.
    """
    print("Application startup: Initializing resources...")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("CRITICAL: OPENAI_API_KEY is not set or is still the placeholder. LLM functionality will fail.")
        # Optionally raise an error or prevent startup if API key is critical
        # raise ValueError("OPENAI_API_KEY is not configured.")

    try:
        redis_client = get_redis_client()
        mongo_db = get_mongo_client() # This is PyMongo DB object, not client itself
        
        # Initialize RediSearch index if it doesn't exist
        # create_redis_search_index might print to console, which is fine for startup.
        create_redis_search_index(redis_client) 

        memory_manager = MemoryManager(redis_client=redis_client, mongo_db=mongo_db)
        
        app_state["redis_client"] = redis_client
        app_state["mongo_db_obj"] = mongo_db # Storing the db object from get_mongo_client()
        app_state["mongo_client"] = mongo_db.client # Storing the actual MongoClient instance for closing
        app_state["memory_manager"] = memory_manager
        print("Resources initialized successfully.")
    except Exception as e:
        print(f"Error during startup: {e}")
        # Depending on the severity, you might want to exit or raise the error
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on application shutdown.
    """
    print("Application shutdown: Cleaning up resources...")
    redis_client = app_state.get("redis_client")
    if redis_client:
        try:
            redis_client.close() # redis.Redis.close() is synchronous
            print("Redis client closed.")
        except Exception as e: # Catch specific exceptions if possible
            print(f"Error closing Redis client: {e}")

    mongo_client = app_state.get("mongo_client") # This is the MongoClient instance
    if mongo_client:
        try:
            mongo_client.close()
            print("MongoDB client closed.")
        except Exception as e:
            print(f"Error closing MongoDB client: {e}")
    print("Resources cleaned up.")

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>AI Memory Chat API</title>
        </head>
        <body>
            <h1>Welcome to the AI Memory Chat API</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation.</p>
        </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request_body: ChatRequest):
    """
    Handles chat requests, interacts with the LLM, and saves conversation history.
    """
    memory_manager = app_state.get("memory_manager")
    if not memory_manager:
        raise HTTPException(status_code=500, detail="Memory manager not initialized.")

    session_id = request_body.session_id or str(uuid4())
    
    # Get or create the LLM chain for this session
    # Note: get_llm_chain might not be thread-safe if it modifies global state
    # or if MemoryManager/clients are not handled correctly for concurrency.
    # For this setup, MemoryManager is shared, which should be fine.
    try:
        llm_chain = get_llm_chain(memory_manager=memory_manager, session_id=session_id)
    except Exception as e:
        # This could be due to issues in get_llm_chain, like the memory_key issue if not fixed
        print(f"Error creating LLM chain for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not initialize chat chain: {str(e)}")

    if not request_body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # Asynchronously run the chain
        # ConversationChain.arun expects the input key (e.g. "message")
        response_text = await llm_chain.arun(message=request_body.message)
        return ChatResponse(session_id=session_id, response=response_text)
    except Exception as e:
        print(f"Error during LLM chain execution for session {session_id}: {e}")
        # Check if OPENAI_API_KEY is the placeholder, which is a common issue.
        if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
            raise HTTPException(status_code=500, detail="OpenAI API key is not configured correctly on the server.")
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")


@app.post("/search_memory", response_model=SearchResponse)
async def search_memory_endpoint(request_body: SearchRequest):
    """
    Searches the conversation memory.
    """
    memory_manager = app_state.get("memory_manager")
    if not memory_manager:
        raise HTTPException(status_code=500, detail="Memory manager not initialized.")

    if not request_body.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        if request_body.search_type == "semantic":
            results = await memory_manager.search_memory(query_text=request_body.query, top_k=request_body.top_k)
        elif request_body.search_type == "full_text":
            results = await memory_manager.full_text_search(query_text=request_body.query, top_k=request_body.top_k)
        else:
            # This case should ideally be caught by Pydantic validation, but as a safeguard:
            raise HTTPException(status_code=400, detail="Invalid search_type. Must be 'semantic' or 'full_text'.")
        
        # Convert results to SearchResultItem instances
        response_items = [SearchResultItem(**item) for item in results]
        return SearchResponse(results=response_items)
        
    except Exception as e:
        print(f"Error during memory search: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching memory: {str(e)}")

@app.get("/memory/{session_id}", response_model=List[Dict[str, Any]])
async def get_session_memory_endpoint(session_id: str):
    """
    Retrieves all messages for a given session_id directly from Redis.
    This is a raw dump and might be extensive.
    """
    memory_manager = app_state.get("memory_manager")
    if not memory_manager:
        raise HTTPException(status_code=500, detail="Memory manager not initialized.")
    
    # The method get_session_history_from_redis returns messages already parsed from JSON
    # and with datetime objects. We might want to serialize datetime for JSON response.
    try:
        session_history = await memory_manager.get_session_history_from_redis(session_id)
        if not session_history:
            raise HTTPException(status_code=404, detail=f"No memory found for session_id: {session_id}")
        
        # Convert datetime objects to ISO format string for JSON serialization
        for message in session_history:
            if 'timestamp' in message and hasattr(message['timestamp'], 'isoformat'):
                message['timestamp'] = message['timestamp'].isoformat()
            # Ensure embedding is list of floats if it exists and is numpy array
            if 'embedding' in message and hasattr(message['embedding'], 'tolist'):
                message['embedding'] = message['embedding'].tolist()

        return session_history
    except HTTPException:
        raise # Re-raise HTTPExceptions (like 404)
    except Exception as e:
        print(f"Error retrieving raw session memory for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving session memory: {str(e)}")

# --- Main Execution (for local development) ---
if __name__ == "__main__":
    print("Starting FastAPI server with uvicorn...")
    # Note: Uvicorn's reload feature might have issues with complex startup/shutdown events
    # especially if they involve non-serializable objects or external connections.
    uvicorn.run(app, host="0.0.0.0", port=8000)
