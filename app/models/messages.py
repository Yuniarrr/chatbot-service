import uuid as uuid_pkg

from enum import Enum
from uuid import UUID
from typing import Union, Optional, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, String, Enum as SQLEnum, String, ForeignKey, Integer
from fastcrud import FastCRUD

from app.core.database import Base
from app.core.schemas import TimestampSchema


class FromMessage(Enum):
    USER = "USER"
    BOT = "BOT"


class Message(Base):
    __tablename__ = "message"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )

    from_message: Mapped[FromMessage] = mapped_column(
        SQLEnum(FromMessage),
        default=FromMessage.USER,
        nullable=False,
    )
    message: Mapped[str] = mapped_column(String, nullable=False)

    file_url: Mapped[str] = mapped_column(String, nullable=True)
    file_name: Mapped[str] = mapped_column(String, nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("UTC"))
    )
    updated_at: Mapped[Union[datetime, None]] = mapped_column(
        DateTime(timezone=True), default=None
    )

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversation.id", ondelete="CASCADE"),
        index=True,
        default=None,
    )
    conversation = relationship(
        "Conversation",
        foreign_keys=[conversation_id],
        back_populates="message",
        lazy="selectin",
    )


####################
# SCHEMA
####################


class MessageBaseModel(BaseModel):
    from_message: Optional[FromMessage] = None
    message: Annotated[
        Union[str, None],
        Field(examples=["content here"]),
    ]
    conversation_id: Annotated[UUID, Field(examples=["id"])]
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None


class MessageModel(TimestampSchema):
    pass


class MessageCreateModel(MessageBaseModel):
    pass


class MessageReadModel(MessageBaseModel):
    id: UUID

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MessageUpdateModel(BaseModel):
    from_message: Optional[FromMessage] = None
    message: Optional[str] = None
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None


class MessageUpdateInternalModel(MessageUpdateModel):
    updated_at: datetime


CRUDMessage = FastCRUD[
    Message,
    MessageBaseModel,
    MessageCreateModel,
    MessageReadModel,
    MessageUpdateModel,
    MessageUpdateInternalModel,
]

messages = CRUDMessage(Message)
