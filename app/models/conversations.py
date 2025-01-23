import uuid as uuid_pkg

from enum import Enum
from typing import Union
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, String, Enum as String, ForeignKey

from app.core.database import Base


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
