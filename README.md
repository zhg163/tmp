# AI Memory Service with Redis Stack, MongoDB, LangChain & FastAPI

## Overview

This project implements an AI-powered chat service with persistent memory capabilities. It leverages Redis Stack for fast caching and advanced search features (RediSearch), MongoDB for long-term storage of conversation histories, LangChain for structuring interactions with Large Language Models (LLMs), and FastAPI for serving the API. The primary goal is to demonstrate a robust memory system that can recall previous interactions and perform semantic searches on conversation history.

## Features

*   **Persistent Memory**: Conversations are saved and can be recalled across sessions.
*   **Fast Retrieval**: Redis Stack is used for quick access to recent conversation history.
*   **Semantic Search**: Allows searching through conversation history based on meaning, not just keywords.
*   **Full-Text Search**: Supports traditional keyword-based search on conversation content.
*   **MongoDB Backend**: Ensures long-term, durable storage of all conversation data.
*   **LangChain Integration**: Utilizes LangChain for managing LLM interactions and memory.
*   **FastAPI Interface**: Provides a modern, asynchronous API for interacting with the service.

## Core Technologies Used

*   **Redis Stack**: For caching, RediSearch (vector search, full-text search).
*   **MongoDB**: For persistent storage of conversation sessions.
*   **LangChain**: Framework for LLM application development, including memory management.
*   **FastAPI**: For building the asynchronous API.
*   **OpenAI**: For accessing LLMs (e.g., GPT-3.5-turbo for chat, text-embedding-ada-002 for embeddings).
*   **Docker**: For running database services.

## Prerequisites

*   Python 3.8+
*   Docker and Docker Compose (or just Docker if starting containers individually).
*   Access to an OpenAI API key.

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory> # Replace <repository_directory> with the cloned folder name
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    *   The project uses a `.env` file to manage environment variables. A file named `.env` should already exist in the project root with the following content (if not, create it):
        ```env
        OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
        REDIS_HOST="localhost"
        REDIS_PORT=6379
        REDIS_DB=0
        MONGO_URI="mongodb://localhost:27017/"
        MONGO_DB_NAME="ai_memory_db"
        MONGO_COLLECTION_SESSIONS="sessions"
        ```
    *   **Crucial**: Replace `"YOUR_OPENAI_API_KEY_HERE"` with your actual OpenAI API key.
    *   The default values for `REDIS_HOST`, `MONGO_URI`, etc., are set for local Docker instances. Modify them if your database setup differs.

## Running the Services (Databases)

You need to have Redis Stack and MongoDB instances running. You can use Docker for this:

1.  **Run Redis Stack**:
    This command starts a Redis Stack container, which includes RediSearch. Port `8001` is for RedisInsight (web UI).
    ```bash
    docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
    ```
    *To persist Redis data across container restarts, you would add a volume mount, e.g., `-v redis_data:/data`.*

2.  **Run MongoDB**:
    This command starts a MongoDB container.
    ```bash
    docker run -d --name mongo -p 27017:27017 mongo:latest
    ```
    *To persist MongoDB data across container restarts, you would add a volume mount, e.g., `-v mongo_data:/data/db`.*

## Running the FastAPI Application

Once the databases are running and environment variables are set:

1.  **Start the FastAPI server**:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload` enables auto-reloading on code changes, useful for development.

2.  **Access API Documentation**:
    Open your browser and go to `http://localhost:8000/docs`. You'll find the Swagger UI for interacting with the API endpoints.

## Testing the Application

You can test the application using the `/docs` UI or tools like `curl` or Postman. Here's a suggested flow:

1.  **First Chat (New Session)**:
    *   Send a POST request to `/chat` with a message. Do not provide a `session_id`.
    *   Example Request Body: `{"message": "Hello, this is my first message!"}`
    *   The API will return a response from the AI and a new `session_id`. Note this `session_id`.

2.  **Continue Chat (Same Session)**:
    *   Send another POST request to `/chat`, this time including the `session_id` received from the previous step.
    *   Example Request Body: `{"session_id": "your_received_session_id", "message": "Can you remember what I said earlier?"}`
    *   The AI should be able to recall information from the previous turn in the same session.

3.  **Test Memory Recovery (Simulate Cache Clear)**:
    *   To simulate a scenario where Redis cache might be cleared (e.g., Redis restart), you can restart the Redis Docker container:
        ```bash
        docker restart redis-stack
        ```
    *   Wait a few seconds for Redis to restart.
    *   Send another message to the *same session* using the `session_id` from step 1:
        Example Request Body: `{"session_id": "your_received_session_id", "message": "Do you still remember our conversation after the restart?"}`
    *   The application should automatically reload the session history from MongoDB into Redis, and the AI should still have context.

4.  **Test Memory Search**:
    *   Send a POST request to `/search_memory`.
    *   **Semantic Search**:
        Example Request Body: `{"query": "first message content", "search_type": "semantic", "top_k": 3}`
    *   **Full-Text Search**:
        Example Request Body: `{"query": "recall", "search_type": "full_text", "top_k": 3}`
    *   Examine the results to see if relevant conversation snippets are returned.

## Running Unit Tests

To run the automated unit tests:

1.  Ensure all development dependencies are installed (including `pytest` and `pytest-asyncio` from `requirements.txt`).
2.  From the project root directory, run:
    ```bash
    pytest
    ```

This will discover and execute all tests in the `tests/` directory.
