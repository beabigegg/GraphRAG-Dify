from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DifyConfig:
    base_url: str = os.getenv("DIFY_BASE_URL", "https://api.dify.ai")
    api_key: str = os.getenv("DIFY_API_KEY", "")
    app_type: str = os.getenv("DIFY_APP_TYPE", "workflow")  # "chat" or "workflow"
    response_mode: str = os.getenv("DIFY_RESPONSE_MODE", "blocking")
    user_id: str = os.getenv("DIFY_USER_ID", "demo-user")

@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "neo4j")

DIFY = DifyConfig()
NEO4J = Neo4jConfig()