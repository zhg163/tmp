from langchain_openai import ChatOpenAI # Corrected import for ChatOpenAI
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate # Corrected import
from langchain_core.messages import SystemMessage # Corrected import

from langchain_utils.custom_memory import CustomRedisMongoMemory
from db.memory_manager import MemoryManager
from config import LLM_MODEL_NAME, OPENAI_API_KEY

def get_llm_chain(memory_manager: MemoryManager, session_id: str) -> ConversationChain:
    """
    Get a configured LangChain ConversationChain with custom memory.
    """
    llm = ChatOpenAI(
        temperature=0.7, 
        model_name=LLM_MODEL_NAME, 
        openai_api_key=OPENAI_API_KEY
    )
    
    # Instantiate the custom memory
    # NB: CustomRedisMongoMemory's __init__ does not accept 'memory_key'.
    # This line, as provided in the subtask, will raise a TypeError.
    # If the intention is to set memory_key, it should be:
    # memory = CustomRedisMongoMemory(...)
    # memory.memory_key = "chat_history"
    # Or CustomRedisMongoMemory should be updated to accept memory_key.
    # For now, proceeding with the exact content as requested.
    memory = CustomRedisMongoMemory(
        memory_manager=memory_manager, 
        session_id=session_id
    )

    # Define the chat prompt template
    # The `memory.memory_key` (which defaults to "history" from BaseChatMemory) must match the variable_name in MessagesPlaceholder.
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a friendly AI assistant, designed to help users answer questions and remember previous conversations."),
        MessagesPlaceholder(variable_name=memory.memory_key), # Placeholder for chat history
        HumanMessagePromptTemplate.from_template("{message}") # Placeholder for the current user message (input_key)
    ])

    # Create the ConversationChain
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False, # Set to True for detailed chain operation logging
        # The input key for the chain should match the variable in HumanMessagePromptTemplate (e.g., "message")
        # This is implicitly handled by ConversationChain if the prompt has a single input variable other than memory.
        # If explicit control is needed: input_key="message"
    )
    return conversation_chain
