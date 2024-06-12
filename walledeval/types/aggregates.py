# walledeval/types/aggregates.py

from pydantic import BaseModel


class MCQAggregate(BaseModel):
    accuracy: float