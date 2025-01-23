import uuid as uuid_pkg
from datetime import datetime
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_serializer


class TimestampSchema(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("UTC")))
    updated_at: datetime = Field(default=None)

    @field_serializer("created_at")
    def serialize_dt(
        self, created_at: Union[datetime, None], _info: Any
    ) -> Union[str, None]:
        if created_at is not None:
            return created_at.isoformat()

        return None

    @field_serializer("updated_at")
    def serialize_updated_at(
        self, updated_at: Union[datetime, None], _info: Any
    ) -> Union[str, None]:
        if updated_at is not None:
            return updated_at.isoformat()

        return None
