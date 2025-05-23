import redis
from pymongo import MongoClient
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, MONGO_URI, MONGO_DB_NAME, EMBEDDING_DIM

# 在应用启动时创建 RediSearch 索引 (如果不存在)
def create_redis_search_index(r: redis.Redis, index_name: str = "idx_session_messages"):
    from redis.commands.search.field import TagField, TextField, VectorField, NumericField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType

    try:
        r.ft(index_name).info()
        print(f"RediSearch index '{index_name}' already exists.")
    except Exception: # Should be redis.exceptions.ResponseError but catching generic Exception for broader compatibility
        print(f"Creating RediSearch index '{index_name}'...")
        schema = (
            TagField("session_id"),
            NumericField("turn_id"),
            TagField("role"),
            TextField("content", as_content=True), # Corrected: as_content=True for TextField
            VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": EMBEDDING_DIM, # Use EMBEDDING_DIM from config
                "DISTANCE_METRIC": "COSINE"
            })
        )
        # Corrected: IndexDefinition prefix should be a list e.g., ["session:"]
        # Corrected: IndexType should be IndexType.JSON
        definition = IndexDefinition(prefix=["session:"], index_type=IndexType.JSON)
        r.ft(index_name).create_index(fields=schema, definition=definition) # Corrected: pass schema as fields
        print(f"RediSearch index '{index_name}' created.")

def get_redis_client():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    # 确保 RediSearch 模块已加载
    try:
        r.ping() # Ping is a good way to check basic connectivity
        # Optionally, check for RediSearch module explicitly if needed, e.g. by trying a RediSearch command
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis. Please ensure Redis Stack is running. Error: {e}")
        # Depending on application requirements, you might raise the error or exit
        raise  # Re-raise the exception to be handled by the caller or FastAPI startup
    return r

def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster') 
        db = client[MONGO_DB_NAME]
        print(f"Successfully connected to MongoDB: {MONGO_URI}, database: {MONGO_DB_NAME}")
        return db
    except Exception as e: # pymongo.errors.ConnectionFailure can be more specific
        print(f"Could not connect to MongoDB. Please ensure MongoDB is running and URI is correct. Error: {e}")
        # Depending on application requirements, you might raise the error or exit
        raise # Re-raise the exception
