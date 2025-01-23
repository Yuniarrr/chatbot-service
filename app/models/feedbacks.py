from enum import Enum
from typing import Union
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import (
    DateTime,
    String,
    Enum as SQLEnum,
    String,
)

from app.core.database import Base


class FeedbackType(Enum):
    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(
        "id",
        autoincrement=True,
        nullable=False,
        unique=True,
        primary_key=True,
    )

    type: Mapped[FeedbackType] = mapped_column(
        SQLEnum(FeedbackType),
        default=FeedbackType.POSITIVE,
        nullable=False,
    )
    message: Mapped[str] = mapped_column(String, nullable=False)
    sender: Mapped[Union[str, None]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("UTC"))
    )
    updated_at: Mapped[Union[datetime, None]] = mapped_column(
        DateTime(timezone=True), default=None
    )
