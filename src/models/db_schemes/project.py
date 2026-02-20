from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectID

class Project(BaseModel):
    _id: Optional[ObjectID]
    project_id: str = Field(..., min_lenght=1)

    # In case I want to validate project_id coming from user
    @validator
    def validate_project_id(cls, value):
        if not value.isalnum():
            raise ValueError("Project ID must be alphanumeric")

        return value

    # To make pydantic ignore deal with unkown types to hime like ObjectID
    class config:
        arbitrary_types_allowed = True
