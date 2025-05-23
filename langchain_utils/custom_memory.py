import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, message_from_dict, messages_to_dict
from langchain.memory.utils import get_prompt_input_key
from langchain.memory.chat_memory import BaseChatMemory # Base class for chat memory

from db.memory_manager import MemoryManager

# Helper function to convert DB message format to Langchain message format
def _convert_db_message_to_langchain(message: Dict[str, Any]) -> BaseMessage:
    role = message.get("role", "unknown").lower()
    content = message.get("content", "")
    if role == "user":
        return HumanMessage(content=content)
    elif role == "ai" or role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        # Fallback for unknown roles, or consider raising an error
        return HumanMessage(content=f"Unknown role ({role}): {content}")


# Helper function to convert Langchain message format to DB message format for saving
# This might not be directly needed if MemoryManager handles saving based on role and content
# but it's good for clarity if we need to pass structured message objects.
def _convert_langchain_message_to_db(message: BaseMessage) -> Dict[str, Any]:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "ai"
    elif isinstance(message, SystemMessage):
        role = "system"
    else:
        role = "unknown" # Should not happen with standard Langchain messages
    return {"role": role, "content": message.content}


class CustomChatMessageHistory(BaseChatMessageHistory):
    """
    Custom chat message history that interacts with MemoryManager.
    This class directly implements the BaseChatMessageHistory interface.
    """
    # Pydantic v2 configuration:
    class Config:
        arbitrary_types_allowed = True

    memory_manager: MemoryManager
    session_id: str
    
    # Keep track of loaded messages to avoid re-fetching within the same instance lifecycle
    # if not strictly necessary by the logic.
    _loaded_messages: Optional[List[BaseMessage]] = None


    def __init__(self, session_id: str, memory_manager: MemoryManager):
        super().__init__()
        self.session_id = session_id
        self.memory_manager = memory_manager
        self._loaded_messages = None # Initialize as not loaded

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve messages from Redis or MongoDB."""
        if self._loaded_messages is None: # Lazy loading
            # print(f"CustomChatMessageHistory: Loading messages for session {self.session_id}")
            history_from_db = asyncio.run(self.memory_manager.get_session_history_from_redis(self.session_id))
            
            if not history_from_db:
                # print(f"CustomChatMessageHistory: No history in Redis for {self.session_id}, trying MongoDB...")
                # Attempt to load from MongoDB into Redis
                loaded_from_mongo = asyncio.run(self.memory_manager.load_session_from_mongo_to_redis(self.session_id))
                if loaded_from_mongo:
                    # print(f"CustomChatMessageHistory: Loaded from MongoDB to Redis, fetching again for {self.session_id}")
                    history_from_db = asyncio.run(self.memory_manager.get_session_history_from_redis(self.session_id))
            
            self._loaded_messages = [_convert_db_message_to_langchain(msg) for msg in history_from_db]
            # print(f"CustomChatMessageHistory: Loaded {len(self._loaded_messages)} messages for session {self.session_id}")
        return self._loaded_messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the store."""
        # print(f"CustomChatMessageHistory: Adding message for session {self.session_id}. Message: {message.type} - {message.content[:50]}")
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage): # Though typically system messages are not added this way during a conversation
            role = "system"
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

        asyncio.run(self.memory_manager.save_message(
            session_id=self.session_id,
            role=role,
            content=message.content
        ))
        # Invalidate the cache to force reload on next access
        self._loaded_messages = None 

    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Clear messages from the store."""
        # This is more complex. Clearing means deleting from Redis and MongoDB.
        # For Redis, one might delete keys matching session_id.
        # For MongoDB, one might remove the messages array or the session document.
        # This depends on the desired behavior of "clear".
        # For now, let's assume it means clearing the Redis cache and Mongo messages for the session.
        # A simpler implementation might just clear the local cache:
        print(f"CustomChatMessageHistory: Clearing messages for session {self.session_id}. (Note: Actual DB clear not fully implemented here for safety).")
        
        # Example: To clear Redis side for this session (be careful with patterns):
        # keys_to_delete = self.memory_manager.redis.keys(f"session:{self.session_id}:message:*")
        # if keys_to_delete:
        #     self.memory_manager.redis.delete(*keys_to_delete)
        
        # Example: To clear MongoDB messages for this session:
        # self.memory_manager.mongo_sessions.update_one(
        #     {"_id": self.session_id},
        #     {"$set": {"messages": []}}
        # )
        self._loaded_messages = []


class CustomRedisMongoMemory(BaseChatMemory):
    """
    Custom LangChain memory class that uses MemoryManager for persistence.
    It uses CustomChatMessageHistory as its chat_memory.
    """
    memory_manager: MemoryManager
    session_id: str # Required session_id

    # Pydantic V2 configuration:
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, 
                 session_id: str, 
                 memory_manager: MemoryManager, 
                 input_key: Optional[str] = None, 
                 output_key: Optional[str] = None,
                 return_messages: bool = False): # Added return_messages for compatibility with BaseChatMemory
        
        # Initialize CustomChatMessageHistory
        chat_history = CustomChatMessageHistory(session_id=session_id, memory_manager=memory_manager)
        
        # Call super with the initialized chat_history
        # Note: BaseChatMemory.__init__ expects chat_memory, input_key, output_key, return_messages
        super().__init__(chat_memory=chat_history, 
                         input_key=input_key, 
                         output_key=output_key, 
                         return_messages=return_messages)
        
        # Store memory_manager and session_id on the instance if needed for other methods,
        # though BaseChatMemory primarily interacts via self.chat_memory
        self.memory_manager = memory_manager 
        self.session_id = session_id
        # print(f"CustomRedisMongoMemory initialized for session {self.session_id}")

    @property
    def memory_variables(self) -> List[str]:
        """Defines the variables this memory will add to the chain input."""
        # This is typically ["history"] for buffer memory.
        return [self.memory_key] # memory_key is 'history' by default in BaseChatMemory


    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables for the chain.
        This method is called by the chain before execution.
        `inputs` is a dictionary of all inputs to the chain.
        """
        # print(f"CustomRedisMongoMemory: Loading memory variables for session {self.session_id}. Inputs: {inputs.keys()}")
        # BaseChatMemory's load_memory_variables handles fetching from self.chat_memory (CustomChatMessageHistory)
        # and formatting it (e.g., as a string or list of messages).
        # The CustomChatMessageHistory.messages property already implements the load-on-demand logic.
        
        # The actual loading logic (Redis -> Mongo -> Redis) is within CustomChatMessageHistory.messages
        # BaseChatMemory.load_memory_variables will call self.chat_memory.messages
        # and then format it using get_buffer_string if return_messages is False.
        
        return super().load_memory_variables(inputs)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save the context of this model run to memory.
        This method is called by the chain after execution.
        """
        # print(f"CustomRedisMongoMemory: Saving context for session {self.session_id}. Inputs: {inputs.keys()}, Outputs: {outputs.keys()}")
        # BaseChatMemory's save_context handles extracting user input and AI output 
        # and adding them to self.chat_memory.
        # It uses self.input_key and self.output_key to find the relevant pieces of information.
        
        # The CustomChatMessageHistory.add_message (called by super().save_context)
        # already handles saving to the MemoryManager.
        super().save_context(inputs, outputs)
        # print(f"CustomRedisMongoMemory: Context saved for session {self.session_id}")

    def clear(self) -> None:
        """Clear memory contents."""
        # print(f"CustomRedisMongoMemory: Clearing memory for session {self.session_id}")
        super().clear() # This will call self.chat_memory.clear()
        # print(f"CustomRedisMongoMemory: Memory cleared for session {self.session_id}")

# Example Usage (Conceptual - would need MemoryManager instance)
# if __name__ == '__main__':
#     # This is for demonstration and would require running Redis/Mongo and a MemoryManager instance
#     # from db.clients import get_redis_client, get_mongo_client
#     # try:
#     #     redis_client = get_redis_client()
#     #     mongo_db = get_mongo_client()
#     #     mm = MemoryManager(redis_client=redis_client, mongo_db=mongo_db)
#     # 
#     #     session_id = "test_session_custom_mem_002"
#     # 
#     #     # Test CustomChatMessageHistory
#     #     history = CustomChatMessageHistory(session_id=session_id, memory_manager=mm)
#     #     print("Initial history messages:", history.messages)
#     #     history.add_user_message("Hello there!")
#     #     history.add_ai_message("Hi! How can I help you today?")
#     #     print("Updated history messages:", history.messages)
#     # 
#     #     # Test CustomRedisMongoMemory
#     #     memory = CustomRedisMongoMemory(session_id=session_id, memory_manager=mm, input_key="input", output_key="output")
#     #     
#     #     # Simulate loading (normally done by the chain)
#     #     mem_vars = memory.load_memory_variables({"input": "some irrelevant input for loading"})
#     #     print("Loaded memory variables:", mem_vars)
#     # 
#     #     # Simulate saving (normally done by the chain)
#     #     memory.save_context({"input": "What is the weather like?"}, {"output": "It is sunny today."})
#     #     
#     #     mem_vars_after_save = memory.load_memory_variables({"input": "again, irrelevant"})
#     #     print("Memory variables after save:", mem_vars_after_save)
#     #
#     #     # history.clear() # Test clearing
#     #     # print("History after clear:", history.messages)
#     #
#     # except redis.exceptions.ConnectionError as e:
#     #     print(f"Redis connection failed: {e}. Skipping example usage.")
#     # except Exception as e:
#     #     print(f"An error occurred during example usage: {e}")
#     pass
