import json
# import time # Not used directly, can be removed
from uuid import uuid4 # Not used directly in this version, can be removed
from datetime import datetime
from typing import List, Dict, Any, Optional

import redis
import numpy as np # Import numpy
from pymongo.database import Database
from openai import OpenAI # Ensure this is OpenAI v1.x.x or later

from config import MONGO_COLLECTION_SESSIONS, EMBEDDING_MODEL_NAME, EMBEDDING_DIM, OPENAI_API_KEY
# from db.clients import get_redis_client, get_mongo_client # Not needed for the class itself

class MemoryManager:
    def __init__(self, redis_client: redis.Redis, mongo_db: Database, openai_api_key: Optional[str] = None):
        self.redis = redis_client
        self.mongo_sessions = mongo_db[MONGO_COLLECTION_SESSIONS]
        # Initialize OpenAI client. If api_key is not provided, it will try to use env var OPENAI_API_KEY
        self.openai_client = OpenAI(api_key=openai_api_key if openai_api_key else OPENAI_API_KEY)
        self.embedding_model = EMBEDDING_MODEL_NAME
        self.search_index_name = "idx_session_messages" # As defined in clients.py

    def _generate_embedding(self, text: str) -> List[float]:
        """Generates embedding for text"""
        if not text or not text.strip(): # Handle empty or whitespace-only strings
            return [0.0] * EMBEDDING_DIM 
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for text '{text[:50]}...': {e}")
            return [0.0] * EMBEDDING_DIM


    async def load_session_from_mongo_to_redis(self, session_id: str) -> bool:
        """
        Loads session data from MongoDB to Redis.
        Called when Redis does not have the corresponding session.
        """
        session_data = self.mongo_sessions.find_one({"_id": session_id})
        if not session_data:
            print(f"Session '{session_id}' not found in MongoDB.")
            return False

        print(f"Loading session '{session_id}' from MongoDB to Redis...")
        pipeline = self.redis.pipeline(transaction=False)
        loaded_messages = 0
        for message in session_data.get("messages", []):
            turn_id = message["turn_id"]
            redis_key = f"session:{session_id}:message:{turn_id}"
            
            timestamp_iso = message["timestamp"]
            if isinstance(timestamp_iso, datetime):
                timestamp_iso = timestamp_iso.isoformat()

            message_to_store = {
                "session_id": session_id,
                "turn_id": turn_id,
                "role": message["role"],
                "content": message["content"],
                "timestamp": timestamp_iso,
                "embedding": message.get("embedding", [0.0] * EMBEDDING_DIM) 
            }
            pipeline.json().set(redis_key, "$", message_to_store)
            loaded_messages += 1
        
        try:
            pipeline.execute()
            print(f"Session '{session_id}' loaded {loaded_messages} messages to Redis.")
            return True
        except Exception as e:
            print(f"Error loading session '{session_id}' to Redis via pipeline: {e}")
            return False

    async def get_session_history_from_redis(self, session_id: str) -> List[Dict[str, Any]]:
        """Gets session history from Redis"""
        keys = sorted(self.redis.keys(f"session:{session_id}:message:*")) 
        if not keys:
            return []

        messages = []
        pipeline = self.redis.pipeline(transaction=False)
        for key in keys:
            pipeline.json().get(key)
        
        raw_messages = pipeline.execute()

        for message_json in raw_messages:
            if message_json: 
                if isinstance(message_json.get("timestamp"), str):
                    try:
                        message_json["timestamp"] = datetime.fromisoformat(message_json["timestamp"])
                    except ValueError:
                        print(f"Warning: Could not parse timestamp string: {message_json.get('timestamp')}")
                messages.append(message_json)
        
        messages.sort(key=lambda x: x.get('turn_id', 0))
        return messages

    async def save_message(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        """
        Saves a single message round to Redis and MongoDB.
        """
        session_doc = self.mongo_sessions.find_one({"_id": session_id}, {"messages": {"$slice": -1}})
        current_turn_id = 1
        if session_doc and "messages" in session_doc and session_doc["messages"]:
            last_turn_id = session_doc["messages"][0].get("turn_id", 0)
            current_turn_id = last_turn_id + 1
        else: 
            existing_session = self.mongo_sessions.find_one({"_id": session_id})
            if existing_session and "messages" in existing_session:
                 current_turn_id = len(existing_session["messages"]) + 1

        timestamp = datetime.now()
        embedding = self._generate_embedding(content)

        message_data_mongo = {
            "turn_id": current_turn_id,
            "role": role,
            "content": content,
            "timestamp": timestamp, 
            "embedding": embedding,
            "extracted_keywords": [] 
        }

        redis_key = f"session:{session_id}:message:{current_turn_id}"
        redis_message_data = {
            "session_id": session_id, 
            "turn_id": current_turn_id,
            "role": role,
            "content": content,
            "timestamp": timestamp.isoformat(), 
            "embedding": embedding
        }
        self.redis.json().set(redis_key, "$", redis_message_data)

        self.mongo_sessions.update_one(
            {"_id": session_id},
            {
                "$push": {"messages": message_data_mongo},
                "$set": {"last_updated_at": timestamp},
                "$setOnInsert": {
                    "user_id": "default_user", 
                    "created_at": timestamp,
                    "title": f"Session started at {timestamp.strftime('%Y-%m-%d %H:%M')}", 
                    "memory_facts": [] 
                }
            },
            upsert=True 
        )
        print(f"Saved message for session {session_id}, turn {current_turn_id}")
        return message_data_mongo 

    async def search_memory(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic search using RediSearch and VECTOR.
        """
        if not query_text.strip():
            return []
            
        query_embedding = self._generate_embedding(query_text)
        
        from redis.commands.search.query import Query
        
        query_vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        q = Query(f"*=>[KNN {top_k} @embedding $query_vector AS vector_score]") \
            .return_fields("session_id", "turn_id", "role", "content", "timestamp", "vector_score") \
            .sort_by("vector_score") \
            .dialect(2)

        query_params = {"query_vector": query_vector_bytes}
        
        try:
            results = self.redis.ft(self.search_index_name).search(q, query_params=query_params)
        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []
        
        found_memories = []
        for doc in results.docs:
            message_data = {
                "session_id": doc.session_id,
                "turn_id": int(doc.turn_id), 
                "role": doc.role,
                "content": doc.content,
                "timestamp": doc.timestamp, 
                "score": float(doc.vector_score) 
            }
            found_memories.append(message_data)
        
        return found_memories

    async def full_text_search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs full-text search using RediSearch"""
        if not query_text.strip():
            return []

        from redis.commands.search.query import Query

        q = Query(query_text) \
            .return_fields("session_id", "turn_id", "role", "content", "timestamp") \
            .paging(0, top_k) \
            .dialect(2)
        
        try:
            results = self.redis.ft(self.search_index_name).search(q)
        except Exception as e:
            print(f"Error during full-text search: {e}")
            return []
            
        found_memories = []
        for doc in results.docs:
            message_data = {
                "session_id": doc.session_id,
                "turn_id": int(doc.turn_id),
                "role": doc.role,
                "content": doc.content,
                "timestamp": doc.timestamp, 
            }
            found_memories.append(message_data)
        return found_memories
