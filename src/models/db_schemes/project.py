from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional
from bson.objectid import ObjectId

class Project(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    project_id: str = Field(..., min_lenght=1)

    # In case I want to validate project_id coming from user
    @validator('project_id')
    def validate_project_id(cls, value):
        if not value.isalnum():
            raise ValueError("Project ID must be alphanumeric")

        return value

    # To make pydantic ignore deal with unkown types to hime like ObjectID
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_indexes(cls):

        return [
            {
                "key": [
                    ("product_id", 1) # 1 --> asending -1 --> desending
                ],
                "name": "product_id_index_1",
                "unique": True
            }
        ]
