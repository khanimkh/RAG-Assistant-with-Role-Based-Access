from typing import List
from pydantic import BaseModel


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    latency: float
