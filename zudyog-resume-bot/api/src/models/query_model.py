from pydantic import BaseModel


class QueryRequest(BaseModel):
    text: str
    temperature: float = 0.1
    threshold: float = 0.3
    namespace: str = "default"
    query_id: str = None


class QueryResponse(BaseModel):
    response: str
    query_id: str
