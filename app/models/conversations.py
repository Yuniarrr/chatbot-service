import uuid as uuid_pkg

from uuid import UUID
from typing import Union, Optional, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, String, Enum as String, ForeignKey
from fastcrud import FastCRUD

from app.core.database import Base
from app.core.schemas import TimestampSchema


class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )

    name: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("UTC"))
    )
    updated_at: Mapped[Union[datetime, None]] = mapped_column(
        DateTime(timezone=True), default=None
    )

    created_by: Mapped[Union[str, None]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        default=None,
        nullable=True,
    )

    user = relationship(
        "User",
        back_populates="conversations",
        lazy="selectin",
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
        lazy="selectin",
        cascade="all, delete",
        passive_deletes=True,
    )


####################
# SCHEMA
####################


class ConversationBaseModel(BaseModel):
    name: Annotated[
        Union[str, None],
        Field(examples=["content here"]),
    ]
    created_by: Optional[UUID] = None


class ConversationModel(TimestampSchema):
    pass


class ConversationCreateModel(ConversationBaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationReadModel(ConversationBaseModel):
    id: UUID

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConversationUpdateModel(BaseModel):
    name: Optional[str] = None


class ConversationUpdateInternalModel(ConversationUpdateModel):
    updated_at: datetime


CRUDConversation = FastCRUD[
    Conversation,
    ConversationBaseModel,
    ConversationCreateModel,
    ConversationReadModel,
    ConversationUpdateModel,
    ConversationUpdateInternalModel,
]

crud_conversation = FastCRUD(CRUDConversation)
