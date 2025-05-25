import uuid as uuid_pkg

from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import DateTime, Text
from typing import Union
from pydantic import BaseModel, ConfigDict
from fastcrud import FastCRUD

from app.core.database import Base
from app.core.schemas import TimestampSchema


class Collection(Base):
    __tablename__ = "collection"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )
    name: Mapped[str] = mapped_column(unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    count: Mapped[int] = mapped_column(default=0, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("UTC"))
    )
    updated_at: Mapped[Union[datetime, None]] = mapped_column(
        DateTime(timezone=True), default=None
    )


####################
# FORMS
####################


class UpdateCollectionForm(BaseModel):
    status: bool


####################
# SCHEMA
####################


class CollectionBaseModel(BaseModel):
    name: str
    description: str
    is_active: bool = True
    count: int = 0


class CollectionModel(TimestampSchema):
    pass


class CollectionCreateModel(CollectionBaseModel):
    model_config = ConfigDict(extra="forbid")


class CollectionReadModel(CollectionBaseModel):
    id: uuid_pkg.UUID
    created_at: datetime
    updated_at: Union[datetime, None]

    class Config:
        from_attributes = True


class CollectionUpdateModel(CollectionBaseModel):
    model_config = ConfigDict(extra="forbid")


class CollectionUpdateInternalModel(CollectionBaseModel):
    updated_at: datetime


CRUDCollection = FastCRUD[
    CollectionBaseModel,
    CollectionModel,
    CollectionReadModel,
    CollectionCreateModel,
    CollectionUpdateModel,
    CollectionUpdateInternalModel,
]

collections = CRUDCollection(Collection)
