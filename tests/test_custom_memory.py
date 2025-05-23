import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

# Corrected import path based on file structure
from langchain_utils.custom_memory import CustomChatMessageHistory, CustomRedisMongoMemory 
from db.memory_manager import MemoryManager # For type hinting and mocking

SAMPLE_SESSION_ID = "test_session_custom_mem"

@pytest.fixture
def mock_memory_manager():
    # MemoryManager's methods are async, so use AsyncMock
    manager = MagicMock(spec=MemoryManager) 
    manager.get_session_history_from_redis = AsyncMock(return_value=[])
    manager.load_session_from_mongo_to_redis = AsyncMock(return_value=True)
    manager.save_message = AsyncMock()
    # Add other methods if they are called and need specific mock behavior
    return manager

@pytest.fixture
def chat_history_backend(mock_memory_manager): # Renamed from chat_history to avoid confusion with memory.chat_history
    # This is CustomChatMessageHistory
    return CustomChatMessageHistory(session_id=SAMPLE_SESSION_ID, memory_manager=mock_memory_manager)

@pytest.fixture
def custom_memory(mock_memory_manager): # This is CustomRedisMongoMemory
    # Provide input_key and output_key as they are used by BaseChatMemory's save_context
    return CustomRedisMongoMemory(
        session_id=SAMPLE_SESSION_ID, 
        memory_manager=mock_memory_manager,
        input_key="user_input", # Example input key
        output_key="ai_response" # Example output key
    )

# --- Tests for CustomChatMessageHistory --- #

@pytest.mark.asyncio
async def test_ch_add_message_human(chat_history_backend: CustomChatMessageHistory, mock_memory_manager: MagicMock):
    message_content = "Hello there user!"
    human_message = HumanMessage(content=message_content)
    
    # add_message calls asyncio.run(self.memory_manager.save_message(...))
    # We need to ensure that save_message is called correctly
    chat_history_backend.add_message(human_message) # This is a synchronous method
    
    # Assert that memory_manager.save_message (an AsyncMock) was called
    mock_memory_manager.save_message.assert_called_once_with(
        session_id=SAMPLE_SESSION_ID,
        role="user",
        content=message_content
    )
    # Adding a message should invalidate the local cache
    assert chat_history_backend._loaded_messages is None

@pytest.mark.asyncio
async def test_ch_add_message_ai(chat_history_backend: CustomChatMessageHistory, mock_memory_manager: MagicMock):
    message_content = "Hello there AI!"
    ai_message = AIMessage(content=message_content)
    
    chat_history_backend.add_message(ai_message)
    
    mock_memory_manager.save_message.assert_called_once_with(
        session_id=SAMPLE_SESSION_ID,
        role="ai",
        content=message_content
    )
    assert chat_history_backend._loaded_messages is None

@pytest.mark.asyncio
async def test_ch_messages_property_initial_load_from_redis(chat_history_backend: CustomChatMessageHistory, mock_memory_manager: MagicMock):
    redis_messages_data = [
        {"role": "user", "content": "Hi from Redis", "turn_id": 1, "timestamp": "sometime"},
        {"role": "ai", "content": "Hello from Redis", "turn_id": 2, "timestamp": "sometime"}
    ]
    mock_memory_manager.get_session_history_from_redis.return_value = redis_messages_data
    
    # Accessing .messages property, which is synchronous but calls asyncio.run internally
    messages = chat_history_backend.messages
    
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hi from Redis"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hello from Redis"
    
    mock_memory_manager.get_session_history_from_redis.assert_called_once_with(SAMPLE_SESSION_ID)
    mock_memory_manager.load_session_from_mongo_to_redis.assert_not_called()
    assert chat_history_backend._loaded_messages == messages # Cache should be populated

@pytest.mark.asyncio
async def test_ch_messages_property_uses_cache_after_initial_load(chat_history_backend: CustomChatMessageHistory, mock_memory_manager: MagicMock):
    redis_messages_data = [{"role": "user", "content": "Cached message"}]
    mock_memory_manager.get_session_history_from_redis.return_value = redis_messages_data
    
    # First access
    first_messages = chat_history_backend.messages
    assert len(first_messages) == 1
    mock_memory_manager.get_session_history_from_redis.assert_called_once() # Called once
    
    # Second access
    second_messages = chat_history_backend.messages
    assert len(second_messages) == 1
    assert second_messages == first_messages # Should be the same cached objects
    mock_memory_manager.get_session_history_from_redis.assert_called_once() # Still called only once

@pytest.mark.asyncio
async def test_ch_messages_property_loads_from_mongo_if_redis_empty(chat_history_backend: CustomChatMessageHistory, mock_memory_manager: MagicMock):
    mongo_loaded_messages_data = [{"role": "ai", "content": "Loaded from Mongo"}]
    
    # Simulate Redis empty, then after Mongo load, Redis has data
    mock_memory_manager.get_session_history_from_redis.side_effect = [
        [], # First call from .messages
        mongo_loaded_messages_data # Second call from .messages after mongo_load attempt
    ]
    mock_memory_manager.load_session_from_mongo_to_redis.return_value = True # Mongo load successful

    messages = chat_history_backend.messages
    
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Loaded from Mongo"
    
    assert mock_memory_manager.get_session_history_from_redis.call_count == 2
    mock_memory_manager.load_session_from_mongo_to_redis.assert_called_once_with(SAMPLE_SESSION_ID)

@pytest.mark.asyncio
async def test_ch_clear(chat_history_backend: CustomChatMessageHistory, mock_memory_manager: MagicMock):
    # Populate cache first
    chat_history_backend._loaded_messages = [HumanMessage(content="test")]
    assert len(chat_history_backend.messages) == 1 # Should use cache
    
    chat_history_backend.clear() # Synchronous method
    
    assert len(chat_history_backend.messages) == 0 # Accessing messages re-triggers load, but clear sets _loaded_messages = []
    assert chat_history_backend._loaded_messages == [] # Cache is cleared
    # Current clear implementation only prints and clears local cache.
    # It does not call memory_manager to delete from DB. If it did, we'd assert those calls.
    # e.g. mock_memory_manager.delete_session_from_redis.assert_called_once()

# --- Tests for CustomRedisMongoMemory --- #

@pytest.mark.asyncio
async def test_cmm_load_memory_vars_integration_with_chat_history(custom_memory: CustomRedisMongoMemory, mock_memory_manager: MagicMock):
    # This test relies on CustomChatMessageHistory's logic, which is tested above.
    # Here, we ensure CustomRedisMongoMemory uses its chat_memory (our CustomChatMessageHistory instance) correctly.
    
    redis_messages_data = [
        {"role": "user", "content": "Input for load_memory_variables", "turn_id": 1},
        {"role": "ai", "content": "Output for load_memory_variables", "turn_id": 2}
    ]
    # Setup the mock_memory_manager that chat_history_backend (CustomChatMessageHistory) uses
    mock_memory_manager.get_session_history_from_redis.return_value = redis_messages_data
    
    # CustomRedisMongoMemory.load_memory_variables is synchronous
    # but it calls CustomChatMessageHistory.messages which calls asyncio.run.
    # The 'inputs' dict is not strictly used by BaseChatMemory.load_memory_variables if input_key/output_key are None
    # or if the primary goal is just to get the history string/messages.
    # For this test, an empty dict is fine as we are focused on history loading.
    memory_vars = custom_memory.load_memory_variables({}) 
    
    # CustomRedisMongoMemory (via BaseChatMemory) by default returns history as a list of BaseMessage objects
    # if return_messages=True (default is False, returns string). Let's re-init for messages.
    custom_memory_for_messages = CustomRedisMongoMemory(
        session_id=SAMPLE_SESSION_ID, 
        memory_manager=mock_memory_manager,
        return_messages=True # Important for this assertion
    )
    
    memory_vars_as_messages = custom_memory_for_messages.load_memory_variables({})

    assert custom_memory_for_messages.chat_memory is not None # chat_memory is CustomChatMessageHistory
    assert len(memory_vars_as_messages[custom_memory_for_messages.memory_key]) == 2
    assert isinstance(memory_vars_as_messages[custom_memory_for_messages.memory_key][0], HumanMessage)
    assert memory_vars_as_messages[custom_memory_for_messages.memory_key][0].content == "Input for load_memory_variables"
    
    # Check that get_session_history_from_redis was called by the underlying CustomChatMessageHistory
    mock_memory_manager.get_session_history_from_redis.assert_called_once_with(SAMPLE_SESSION_ID)

@pytest.mark.asyncio
async def test_cmm_save_context(custom_memory: CustomRedisMongoMemory):
    # custom_memory.chat_memory is an instance of CustomChatMessageHistory
    # We mock its add_message method to check if BaseChatMemory.save_context calls it correctly.
    custom_memory.chat_memory.add_message = MagicMock() # Mocking the sync add_message
    
    # User input_key="user_input", AI output_key="ai_response" from fixture
    inputs = {"user_input": "User says hi to context"}
    outputs = {"ai_response": "AI says hello to context"}
    
    # save_context is synchronous
    custom_memory.save_context(inputs, outputs)
    
    # BaseChatMemory.save_context should call chat_memory.add_user_message then chat_memory.add_ai_message
    # which in turn call chat_memory.add_message
    assert custom_memory.chat_memory.add_message.call_count == 2
    
    calls = custom_memory.chat_memory.add_message.call_args_list
    
    # Call 1 (HumanMessage)
    args_human, _ = calls[0]
    assert isinstance(args_human[0], HumanMessage)
    assert args_human[0].content == "User says hi to context"
    
    # Call 2 (AIMessage)
    args_ai, _ = calls[1]
    assert isinstance(args_ai[0], AIMessage)
    assert args_ai[0].content == "AI says hello to context"

@pytest.mark.asyncio
async def test_cmm_clear(custom_memory: CustomRedisMongoMemory):
    # Mock the clear method of the underlying chat_memory (CustomChatMessageHistory instance)
    custom_memory.chat_memory.clear = MagicMock() # Mocking the sync clear
    
    custom_memory.clear() # Synchronous method
    
    custom_memory.chat_memory.clear.assert_called_once()

```
