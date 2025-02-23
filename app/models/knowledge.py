import logging
import uuid as uuid_pkg

from enum import Enum
from typing import Union, List, Optional, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    DateTime,
    Text,
    ForeignKey,
    JSON,
)
from fastcrud import FastCRUD

from app.core.database import Base, JSONField
from app.models.conversations import Conversation
from app.core.schemas import TimestampSchema
from app.core.logger import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODEL"])


class Knowledge(Base):
    __tablename__ = "knowledge"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    data: Mapped[str] = mapped_column(JSON, nullable=False)
    meta: Mapped[str] = mapped_column(JSON, nullable=False)

    user_id: Mapped[Union[str, None]] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )

    uploader = relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="uploader_knowledge",
        lazy="selectin",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("UTC"))
    )
    updated_at: Mapped[Union[datetime, None]] = mapped_column(
        DateTime(timezone=True), default=None
    )
