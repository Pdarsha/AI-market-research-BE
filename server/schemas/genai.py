from typing import Optional

from pydantic import BaseModel


class GenAIBase(BaseModel):
    user_query: str

    class Config:
        from_attributes = True


class GenAICreate(GenAIBase):
    ai_response: Optional[str] = None


class GenAIUpdate(GenAIBase):
    pass


class AIResponse(BaseModel):
    user_query: str
    response: str


class EmbeddingBase(BaseModel):
    text: str

    class Config:
        from_attributes = True


class CreateEmbedding(EmbeddingBase):
    pass


class updateEmbedding(EmbeddingBase):
    pass


class EmbeddingResponse(BaseModel):
    file_name: str
