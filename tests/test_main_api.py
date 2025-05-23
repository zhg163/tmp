import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from uuid import UUID # For checking session_id format

# --- Global Mocks ---
# These mocks will be applied where 'main.py' imports or uses these names.

# 1. Mock config.OPENAI_API_KEY (used in main.startup_event)
# We'll patch 'main.OPENAI_API_KEY' as it's imported like 'from config import OPENAI_API_KEY' in main.py
# Alternatively, patching 'config.OPENAI_API_KEY' before main is imported also works.
# For simplicity with the fixture structure, patching 'main.OPENAI_API_KEY' is more direct.

# 2. Mock functions from db.clients (used in main.startup_event)
mock_redis_client_instance = MagicMock(name="MockRedisClientInstance")
mock_mongo_client_instance = MagicMock(name="MockMongoClientInstance") # This is the actual client
mock_mongo_db_obj_instance = MagicMock(name="MockMongoDbObjectInstance") # This is db object: client[db_name]
mock_mongo_db_obj_instance.client = mock_mongo_client_instance # Link client to db object for shutdown

mock_get_redis_func = MagicMock(return_value=mock_redis_client_instance, name="MockGetRedisFunc")
mock_get_mongo_func = MagicMock(return_value=mock_mongo_db_obj_instance, name="MockGetMongoFunc")
mock_create_index_func = MagicMock(name="MockCreateIndexFunc")

# 3. Mock MemoryManager class and its instance (used in main.startup_event and endpoints)
# The instance is created in startup_event. We mock the class to control its instance.
mock_memory_manager_instance = AsyncMock(name="MockMemoryManagerInstance")
MockMemoryManagerClass = MagicMock(return_value=mock_memory_manager_instance, name="MockMemoryManagerClass")

# 4. Mock get_llm_chain function and its returned chain instance (used in /chat endpoint)
mock_llm_chain_instance = AsyncMock(name="MockLLMChainInstance")
mock_llm_chain_instance.arun = AsyncMock(return_value="AI Test Response", name="MockArunMethod")
mock_get_llm_chain_func = MagicMock(return_value=mock_llm_chain_instance, name="MockGetLLMChainFunc")


# --- Pytest Fixtures ---
@pytest.fixture(scope="module", autouse=True)
def apply_module_patches():
    """
    Apply patches at the module level. These mocks will be active for all tests.
    Patches are applied where the names are looked up in 'main.py'.
    """
    # Patching config directly as it's imported early.
    with patch('config.OPENAI_API_KEY', "fake_test_key"), \
         patch('main.get_redis_client', mock_get_redis_func), \
         patch('main.get_mongo_client', mock_get_mongo_func), \
         patch('main.create_redis_search_index', mock_create_index_func), \
         patch('main.MemoryManager', MockMemoryManagerClass), \
         patch('main.get_llm_chain', mock_get_llm_chain_func):
        yield

# Import the FastAPI app *after* the module-level patches are set to be applied.
from main import app

@pytest.fixture
def client():
    """
    Provides a TestClient instance for making requests to the FastAPI app.
    Also resets mocks before each test to ensure test isolation.
    """
    # Reset mocks that might carry state between tests
    mock_get_redis_func.reset_mock(return_value=mock_redis_client_instance)
    mock_get_mongo_func.reset_mock(return_value=mock_mongo_db_obj_instance)
    mock_create_index_func.reset_mock()
    
    MockMemoryManagerClass.reset_mock(return_value=mock_memory_manager_instance)
    mock_memory_manager_instance.reset_mock() # Resets call counts etc.
    # Re-configure default async method mocks on the instance after reset
    mock_memory_manager_instance.search_memory = AsyncMock(return_value=[], name="SearchMemoryAsyncMock")
    mock_memory_manager_instance.full_text_search = AsyncMock(return_value=[], name="FullTextSearchAsyncMock")
    mock_memory_manager_instance.get_session_history_from_redis = AsyncMock(return_value=[], name="GetHistoryAsyncMock")
    
    mock_get_llm_chain_func.reset_mock(return_value=mock_llm_chain_instance)
    mock_llm_chain_instance.reset_mock()
    mock_llm_chain_instance.arun = AsyncMock(return_value="AI Test Response", name="ArunAsyncMockPostReset")

    # The app's startup event (which uses these mocks) runs when TestClient is created.
    with TestClient(app) as test_client:
        yield test_client

# --- Test Cases ---

def test_read_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "<h1>Welcome to the AI Memory Chat API</h1>" in response.text # Check for HTML content

def test_chat_new_session(client: TestClient):
    mock_llm_chain_instance.arun.return_value = "Response for new session"
    
    response = client.post("/chat", json={"message": "Hello, new world!"})
    data = response.json()
    
    assert response.status_code == 200
    assert "session_id" in data
    generated_session_id = data["session_id"]
    try:
        UUID(generated_session_id, version=4)
    except ValueError:
        pytest.fail("session_id is not a valid UUID4")
    assert data["response"] == "Response for new session"
    
    # Check that get_llm_chain was called with the MemoryManager instance and a generated session_id
    mock_get_llm_chain_func.assert_called_once()
    args_llm_chain, _ = mock_get_llm_chain_func.call_args
    assert args_llm_chain[0] is mock_memory_manager_instance # Check MemoryManager instance
    assert args_llm_chain[1] == generated_session_id # Check session_id used for chain
    
    mock_llm_chain_instance.arun.assert_called_once_with(message="Hello, new world!")

def test_chat_existing_session(client: TestClient):
    existing_session_id = "my_special_session_id"
    mock_llm_chain_instance.arun.return_value = "Response for existing session"
    
    response = client.post("/chat", json={"session_id": existing_session_id, "message": "Hello again!"})
    data = response.json()
    
    assert response.status_code == 200
    assert data["session_id"] == existing_session_id
    assert data["response"] == "Response for existing session"
    
    mock_get_llm_chain_func.assert_called_once_with(mock_memory_manager_instance, existing_session_id)
    mock_llm_chain_instance.arun.assert_called_once_with(message="Hello again!")

def test_chat_llm_error(client: TestClient):
    mock_llm_chain_instance.arun.side_effect = Exception("LLM processing failed!")
    
    response = client.post("/chat", json={"message": "Trigger error"})
    
    assert response.status_code == 500
    assert "Error processing chat message: LLM processing failed!" in response.json()["detail"]

def test_search_memory_semantic(client: TestClient):
    query = "find semantic stuff"
    mock_results_data = [
        {"session_id": "s1", "turn_id": 1, "role": "user", "content": "semantic result", "timestamp": "ts1", "score": 0.95}
    ]
    mock_memory_manager_instance.search_memory = AsyncMock(return_value=mock_results_data)
    
    response = client.post("/search_memory", json={"query": query, "search_type": "semantic", "top_k": 3})
    data = response.json()
    
    assert response.status_code == 200
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "semantic result"
    mock_memory_manager_instance.search_memory.assert_called_once_with(query_text=query, top_k=3)

def test_search_memory_full_text(client: TestClient):
    query = "find full_text stuff"
    mock_results_data = [
        {"session_id": "s2", "turn_id": 2, "role": "ai", "content": "full_text result", "timestamp": "ts2"}
    ]
    # Ensure this mock is fresh or specific to this test
    mock_memory_manager_instance.full_text_search = AsyncMock(return_value=mock_results_data) 
    
    response = client.post("/search_memory", json={"query": query, "search_type": "full_text", "top_k": 7})
    data = response.json()
    
    assert response.status_code == 200
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "full_text result"
    mock_memory_manager_instance.full_text_search.assert_called_once_with(query_text=query, top_k=7)

def test_search_memory_invalid_type(client: TestClient):
    response = client.post("/search_memory", json={"query": "test", "search_type": "invalid_search"})
    # FastAPI/Pydantic validation error
    assert response.status_code == 422 

def test_get_session_memory(client: TestClient):
    session_id = "session_to_get"
    history_data = [
        {"session_id": session_id, "turn_id": 1, "role": "user", "content": "History item 1", "timestamp": "ts1"},
        {"session_id": session_id, "turn_id": 2, "role": "ai", "content": "History item 2", "timestamp": "ts2"}
    ]
    mock_memory_manager_instance.get_session_history_from_redis = AsyncMock(return_value=history_data)
    
    response = client.get(f"/memory/{session_id}")
    data = response.json()
    
    assert response.status_code == 200
    assert len(data) == 2
    assert data[0]["content"] == "History item 1"
    mock_memory_manager_instance.get_session_history_from_redis.assert_called_once_with(session_id)

def test_get_session_memory_not_found(client: TestClient):
    session_id = "non_existent_session"
    mock_memory_manager_instance.get_session_history_from_redis = AsyncMock(return_value=[]) # No history
    
    response = client.get(f"/memory/{session_id}")
    
    assert response.status_code == 404
    assert response.json()["detail"] == f"No memory found for session_id: {session_id}"
    mock_memory_manager_instance.get_session_history_from_redis.assert_called_once_with(session_id)

# Test startup event dependency initialization
def test_startup_event_initializes_dependencies(client: TestClient):
    # The client fixture itself triggers the startup event.
    # We check if our mocks were called during that startup process.
    mock_get_redis_func.assert_called_once()
    mock_get_mongo_func.assert_called_once()
    mock_create_index_func.assert_called_once_with(mock_redis_client_instance)
    MockMemoryManagerClass.assert_called_once_with(
        redis_client=mock_redis_client_instance, 
        mongo_db=mock_mongo_db_obj_instance
    )
    # Check if app_state in main.py was populated (indirectly, by ensuring mocks were used)
    # from main import app_state # Can be imported to check, but ensure it's after patches
    # assert app_state["redis_client"] is mock_redis_client_instance
    # assert app_state["memory_manager"] is mock_memory_manager_instance
    # This kind of check is good but requires careful handling of import orders and when app_state is set.
    # For now, asserting that the factory functions and class constructors were called is sufficient.
```
