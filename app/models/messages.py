import uuid as uuid_pkg

from enum import Enum
from typing import Union
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, String, Enum as SQLEnum, String, ForeignKey, Integer

from app.core.database import Base


class FromMessage(Enum):
    USER = "USER"
    BOT = "BOT"


class Message(Base):
    __tablename__ = "messages"

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
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
        default=None,
    )
    conversation = relationship(
        "Conversation",
        back_populates="messages",
        lazy="selectin",
    )
