from typing import Optional
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    # No user_role field, deliberately. Role now comes from the verified
    # JWT (see api/dependencies.py's get_current_user), never from
    # something the client can just type into the request body.
