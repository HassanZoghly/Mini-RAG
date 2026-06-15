from token import OP
from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str = None
    # Token-based chunk sizing (see utils.tokenizer.count_tokens).
    # 700-1000 tokens per chunk with 100-150 tokens of overlap keeps
    # enough context for retrieval while still allowing focused answers.
    chunk_size: Optional[int] = 800
    overlap_size: Optional[int] = 125
    do_reset: Optional[int] = 0
