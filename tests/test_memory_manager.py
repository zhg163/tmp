import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call # Added call for checking multiple calls
from datetime import datetime, timezone # Added timezone
from typing import List, Dict, Any

from pymongo.database import Database
from redis import Redis # Actual Redis for spec
from redis.commands.search.document import Document # For mocking search results

from db.memory_manager import MemoryManager
from config import EMBEDDING_DIM, MONGO_COLLECTION_SESSIONS # Import collection name

# --- Constants for Sample Data ---
SAMPLE_SESSION_ID = "test_session_123"
NOW_DATETIME = datetime.now(timezone.utc) # Use timezone-aware datetime
NOW_ISO = NOW_DATETIME.isoformat()
DEFAULT_EMBEDDING = [0.1] * EMBEDDING_DIM
USER_CONTENT = "Hello, this is a user message."
AI_CONTENT = "Hello, this is an AI response."

SAMPLE_MONGO_MESSAGE_USER = {
    "_id": SAMPLE_SESSION_ID, # Example: session document might have an _id
    "messages": [{
        "turn_id": 1,
        "role": "user",
        "content": USER_CONTENT,
        "timestamp": NOW_DATETIME, # Store as datetime object
        "embedding": DEFAULT_EMBEDDING,
        "extracted_keywords": []
    }]
}

SAMPLE_REDIS_MESSAGE_USER = {
    "session_id": SAMPLE_SESSION_ID,
    "turn_id": 1,
    "role": "user",
    "content": USER_CONTENT,
    "timestamp": NOW_ISO, # Store as ISO format string
    "embedding": DEFAULT_EMBEDDING
}

# --- Pytest Fixtures ---
@pytest.fixture
def mock_redis_client():
    client = MagicMock(spec=Redis) # Use spec for better mocking
    
    # Mock json operations
    client.json = MagicMock()
    client.json().set = MagicMock()
    client.json().get = MagicMock(return_value=SAMPLE_REDIS_MESSAGE_USER) # For single get if used

    # Mock keys operation
    client.keys = MagicMock(return_value=[f"session:{SAMPLE_SESSION_ID}:message:1".encode('utf-8')])

    # Mock pipeline for get_session_history_from_redis and load_session_from_mongo_to_redis
    mock_pipeline_obj = MagicMock()
    mock_pipeline_obj.json = MagicMock()
    mock_pipeline_obj.json().get = MagicMock(return_value=SAMPLE_REDIS_MESSAGE_USER)
    mock_pipeline_obj.json().set = MagicMock() # For loading to Redis
    # For get_session_history, execute() returns a list of results from pipeline.json().get() calls
    mock_pipeline_obj.execute = MagicMock(return_value=[SAMPLE_REDIS_MESSAGE_USER]) 
    client.pipeline = MagicMock(return_value=mock_pipeline_obj)
    
    # Mock RediSearch ft().search() and ft().info()
    mock_ft_obj = MagicMock()
    mock_search_result = MagicMock()
    # Simulate a search document structure
    search_doc = Document(id=f"session:{SAMPLE_SESSION_ID}:message:1", payload=None)
    search_doc.session_id = SAMPLE_SESSION_ID
    search_doc.turn_id = 1
    search_doc.role = "user"
    search_doc.content = USER_CONTENT
    search_doc.timestamp = NOW_ISO
    search_doc.vector_score = 0.9 # Example score for semantic search
    
    mock_search_result.docs = [search_doc] 
    mock_ft_obj.search = MagicMock(return_value=mock_search_result)
    mock_ft_obj.info = MagicMock() # For index check, no specific return needed unless testing its output
    client.ft = MagicMock(return_value=mock_ft_obj)
    
    return client

@pytest.fixture
def mock_mongo_db():
    db = MagicMock(spec=Database)
    # The collection mock itself, methods like find_one, update_one will be attached here
    db[MONGO_COLLECTION_SESSIONS] = MagicMock() 
    return db

@pytest.fixture
def mock_openai_client_fixture(): # Renamed to avoid conflict with openai library's client
    client = MagicMock()
    embedding_response = MagicMock()
    embedding_data = MagicMock()
    embedding_data.embedding = [0.123] * EMBEDDING_DIM # Specific embedding for testing
    embedding_response.data = [embedding_data]
    client.embeddings = MagicMock() # Ensure embeddings attribute is a mock
    client.embeddings.create = MagicMock(return_value=embedding_response)
    return client

@pytest.fixture
def memory_manager(mock_redis_client, mock_mongo_db, mock_openai_client_fixture):
    # Use a known fake API key for tests
    mm = MemoryManager(redis_client=mock_redis_client, mongo_db=mock_mongo_db, openai_api_key="fake_testing_key")
    # Directly replace the openai_client instance with our mock
    mm.openai_client = mock_openai_client_fixture 
    return mm

# --- Test Cases --- #

# Test _generate_embedding
@pytest.mark.asyncio
async def test_generate_embedding_valid_text(memory_manager: MemoryManager, mock_openai_client_fixture):
    text = "Hello, world!"
    expected_embedding = [0.123] * EMBEDDING_DIM # Matches mock_openai_client_fixture
    
    embedding = memory_manager._generate_embedding(text) # This is a synchronous method
    
    assert embedding == expected_embedding
    mock_openai_client_fixture.embeddings.create.assert_called_once_with(input=text, model=memory_manager.embedding_model)

@pytest.mark.asyncio
async def test_generate_embedding_empty_text(memory_manager: MemoryManager, mock_openai_client_fixture):
    text = ""
    expected_embedding = [0.0] * EMBEDDING_DIM # Default for empty
    
    embedding = memory_manager._generate_embedding(text)
    
    assert embedding == expected_embedding
    mock_openai_client_fixture.embeddings.create.assert_not_called()

# Test save_message
@pytest.mark.asyncio
async def test_save_message_new_session(memory_manager: MemoryManager, mock_redis_client, mock_mongo_db):
    session_id = "new_test_session_001"
    role = "user"
    content = "A new message for a new session"
    expected_embedding = [0.555] * EMBEDDING_DIM

    # Mock MongoDB find_one for turn_id calculation (return None for new session)
    memory_manager.mongo_sessions.find_one.return_value = None 
    # Mock _generate_embedding for this test
    memory_manager._generate_embedding = MagicMock(return_value=expected_embedding)

    await memory_manager.save_message(session_id, role, content)

    # Assert Redis call
    expected_redis_key = f"session:{session_id}:message:1" # turn_id should be 1
    mock_redis_client.json().set.assert_called_once()
    args, kwargs = mock_redis_client.json().set.call_args
    assert args[0] == expected_redis_key
    assert args[1] == "$" # JSONPath
    redis_data = args[2]
    assert redis_data["session_id"] == session_id
    assert redis_data["turn_id"] == 1
    assert redis_data["role"] == role
    assert redis_data["content"] == content
    assert redis_data["embedding"] == expected_embedding
    # timestamp is tricky to assert exactly, check it's there

    # Assert MongoDB call
    memory_manager.mongo_sessions.update_one.assert_called_once()
    mongo_args, mongo_kwargs = memory_manager.mongo_sessions.update_one.call_args
    assert mongo_args[0] == {"_id": session_id} # Filter
    update_doc = mongo_args[1]
    assert update_doc["$set"]["last_updated_at"] is not None
    assert update_doc["$setOnInsert"]["user_id"] == "default_user"
    pushed_message = update_doc["$push"]["messages"]
    assert pushed_message["turn_id"] == 1
    assert pushed_message["role"] == role
    assert pushed_message["content"] == content
    assert pushed_message["embedding"] == expected_embedding
    assert mongo_kwargs["upsert"] is True

    memory_manager._generate_embedding.assert_called_once_with(content)

@pytest.mark.asyncio
async def test_save_message_existing_session(memory_manager: MemoryManager, mock_redis_client, mock_mongo_db):
    session_id = SAMPLE_SESSION_ID
    role = "ai"
    content = "An AI response in an existing session"
    expected_embedding = [0.666] * EMBEDDING_DIM
    
    # Mock MongoDB find_one for turn_id calculation (return last message for existing session)
    memory_manager.mongo_sessions.find_one.return_value = {"messages": [SAMPLE_MONGO_MESSAGE_USER["messages"][0]]}
    memory_manager._generate_embedding = MagicMock(return_value=expected_embedding)

    await memory_manager.save_message(session_id, role, content)

    expected_turn_id = SAMPLE_MONGO_MESSAGE_USER["messages"][0]["turn_id"] + 1 # Next turn
    expected_redis_key = f"session:{session_id}:message:{expected_turn_id}"
    
    mock_redis_client.json().set.assert_called_once()
    args, _ = mock_redis_client.json().set.call_args
    assert args[0] == expected_redis_key
    redis_data = args[2]
    assert redis_data["turn_id"] == expected_turn_id
    assert redis_data["role"] == role
    assert redis_data["content"] == content

    memory_manager.mongo_sessions.update_one.assert_called_once()
    mongo_args, _ = memory_manager.mongo_sessions.update_one.call_args
    assert mongo_args[0] == {"_id": session_id}
    pushed_message = mongo_args[1]["$push"]["messages"]
    assert pushed_message["turn_id"] == expected_turn_id
    assert pushed_message["role"] == role

    memory_manager._generate_embedding.assert_called_once_with(content)

# Test load_session_from_mongo_to_redis
@pytest.mark.asyncio
async def test_load_session_from_mongo_to_redis_exists(memory_manager: MemoryManager, mock_redis_client, mock_mongo_db):
    # Mongo returns a session with one message
    memory_manager.mongo_sessions.find_one.return_value = SAMPLE_MONGO_MESSAGE_USER 
    
    result = await memory_manager.load_session_from_mongo_to_redis(SAMPLE_SESSION_ID)
    assert result is True
    
    memory_manager.mongo_sessions.find_one.assert_called_once_with({"_id": SAMPLE_SESSION_ID})
    
    # Check if pipeline was used correctly
    mock_redis_client.pipeline.assert_called_once_with(transaction=False)
    pipeline_mock = mock_redis_client.pipeline()
    # One message in SAMPLE_MONGO_MESSAGE_USER
    pipeline_mock.json().set.assert_called_once() 
    args, _ = pipeline_mock.json().set.call_args
    # args[0] is key, args[1] is path, args[2] is data
    assert args[0] == f"session:{SAMPLE_SESSION_ID}:message:{SAMPLE_MONGO_MESSAGE_USER['messages'][0]['turn_id']}"
    assert args[2]['content'] == SAMPLE_MONGO_MESSAGE_USER['messages'][0]['content']
    pipeline_mock.execute.assert_called_once()

@pytest.mark.asyncio
async def test_load_session_from_mongo_to_redis_not_exists(memory_manager: MemoryManager, mock_mongo_db, mock_redis_client):
    memory_manager.mongo_sessions.find_one.return_value = None # Session not in MongoDB
    
    result = await memory_manager.load_session_from_mongo_to_redis("unknown_session")
    assert result is False
    
    memory_manager.mongo_sessions.find_one.assert_called_once_with({"_id": "unknown_session"})
    mock_redis_client.pipeline().json().set.assert_not_called() # No data to set

# Test get_session_history_from_redis
@pytest.mark.asyncio
async def test_get_session_history_from_redis_exists(memory_manager: MemoryManager, mock_redis_client):
    # mock_redis_client.keys already set to return one key
    # mock_redis_client.pipeline().execute already set to return one message
    
    messages = await memory_manager.get_session_history_from_redis(SAMPLE_SESSION_ID)
    
    assert len(messages) == 1
    assert messages[0]["content"] == SAMPLE_REDIS_MESSAGE_USER["content"]
    assert messages[0]["turn_id"] == SAMPLE_REDIS_MESSAGE_USER["turn_id"]
    # Check that timestamp string was converted to datetime object
    assert isinstance(messages[0]["timestamp"], datetime)

    mock_redis_client.keys.assert_called_once_with(f"session:{SAMPLE_SESSION_ID}:message:*")
    mock_redis_client.pipeline().json().get.assert_called_once()
    mock_redis_client.pipeline().execute.assert_called_once()

@pytest.mark.asyncio
async def test_get_session_history_from_redis_empty(memory_manager: MemoryManager, mock_redis_client):
    mock_redis_client.keys.return_value = [] # No keys found in Redis
    
    messages = await memory_manager.get_session_history_from_redis("empty_session")
    
    assert len(messages) == 0
    mock_redis_client.keys.assert_called_once_with(f"session:empty_session:message:*")
    mock_redis_client.pipeline().json().get.assert_not_called()

# Test search_memory (semantic)
@pytest.mark.asyncio
async def test_search_memory_semantic_results(memory_manager: MemoryManager, mock_redis_client):
    query_text = "search for this"
    expected_embedding = [0.789] * EMBEDDING_DIM
    memory_manager._generate_embedding = MagicMock(return_value=expected_embedding)
    
    # mock_redis_client.ft().search is already configured to return one doc
    
    results = await memory_manager.search_memory(query_text, top_k=1)
    
    assert len(results) == 1
    assert results[0]["content"] == USER_CONTENT
    assert results[0]["score"] == 0.9 # From mock_search_result setup
    
    memory_manager._generate_embedding.assert_called_once_with(query_text)
    mock_redis_client.ft(memory_manager.search_index_name).search.assert_called_once()
    # More detailed assertion on query possible here

@pytest.mark.asyncio
async def test_search_memory_semantic_no_results(memory_manager: MemoryManager, mock_redis_client):
    query_text = "search for nothing"
    memory_manager._generate_embedding = MagicMock(return_value=[0.456] * EMBEDDING_DIM)
    
    mock_search_result_empty = MagicMock()
    mock_search_result_empty.docs = [] # No results
    mock_redis_client.ft(memory_manager.search_index_name).search.return_value = mock_search_result_empty
    
    results = await memory_manager.search_memory(query_text, top_k=1)
    
    assert len(results) == 0
    memory_manager._generate_embedding.assert_called_once_with(query_text)

@pytest.mark.asyncio
async def test_search_memory_empty_query(memory_manager: MemoryManager):
    results = await memory_manager.search_memory("")
    assert len(results) == 0
    memory_manager._generate_embedding = MagicMock() # To check it's not called
    memory_manager._generate_embedding.assert_not_called()


# Test full_text_search
@pytest.mark.asyncio
async def test_full_text_search_results(memory_manager: MemoryManager, mock_redis_client):
    query_text = "full text search query"
    # mock_redis_client.ft().search is already configured to return one doc
    # For full-text, score might not be 'vector_score', but from FT result. Modify if needed.
    # The current mock doc has vector_score, which is fine for structure.
    
    results = await memory_manager.full_text_search(query_text, top_k=1)
    
    assert len(results) == 1
    assert results[0]["content"] == USER_CONTENT
    
    mock_redis_client.ft(memory_manager.search_index_name).search.assert_called_once()
    # Check the query object passed to search for full-text

@pytest.mark.asyncio
async def test_full_text_search_no_results(memory_manager: MemoryManager, mock_redis_client):
    query_text = "search for nothing full text"
    
    mock_search_result_empty = MagicMock()
    mock_search_result_empty.docs = [] # No results
    mock_redis_client.ft(memory_manager.search_index_name).search.return_value = mock_search_result_empty
    
    results = await memory_manager.full_text_search(query_text, top_k=1)
    
    assert len(results) == 0

@pytest.mark.asyncio
async def test_full_text_search_empty_query(memory_manager: MemoryManager, mock_redis_client):
    results = await memory_manager.full_text_search("")
    assert len(results) == 0
    mock_redis_client.ft(memory_manager.search_index_name).search.assert_not_called()

# Example of how to mock MongoDB find_one to return different values for different calls or filters
@pytest.mark.asyncio
async def test_save_message_turn_id_logic_detailed(memory_manager: MemoryManager, mock_redis_client, mock_mongo_db):
    session_id = "turn_test_session"
    role = "user"
    content1 = "First message"
    content2 = "Second message"
    
    # Mock _generate_embedding globally for this test
    memory_manager._generate_embedding = MagicMock(return_value=DEFAULT_EMBEDDING)

    # Scenario 1: New session
    mock_mongo_db[MONGO_COLLECTION_SESSIONS].find_one.return_value = None
    await memory_manager.save_message(session_id, role, content1)
    
    args_call1, _ = mock_redis_client.json().set.call_args
    assert args_call1[2]["turn_id"] == 1 # First message is turn 1
    
    # Scenario 2: Existing session, first message saved, now saving second
    # Mock find_one to return the state *after* first message was saved (for turn_id calc)
    # The actual save updates mongo, so next find_one would see it.
    # Here, we simulate what find_one for turn_id calculation would return.
    # This mock simulates that a message with turn_id 1 exists.
    mock_mongo_db[MONGO_COLLECTION_SESSIONS].find_one.return_value = {
        "_id": session_id, 
        "messages": [{"turn_id": 1, "role": "user", "content": content1}]
    }
    await memory_manager.save_message(session_id, role, content2)
    
    # Check the second call to redis.json().set
    args_call2, _ = mock_redis_client.json().set.call_args
    assert args_call2[2]["turn_id"] == 2 # Second message is turn 2

    # Ensure find_one was called twice (once for each save_message)
    assert mock_mongo_db[MONGO_COLLECTION_SESSIONS].find_one.call_count == 2
    # Ensure update_one was called twice
    assert mock_mongo_db[MONGO_COLLECTION_SESSIONS].update_one.call_count == 2
    # Ensure _generate_embedding was called twice
    assert memory_manager._generate_embedding.call_count == 2
```
