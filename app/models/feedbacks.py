import enum
import uuid as uuid_pkg

from uuid import UUID
from typing import Union, Optional, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import (
    DateTime,
    String,
    Enum as SQLEnum,
    String,
)
from fastcrud import FastCRUD

from app.core.database import Base
from app.core.schemas import TimestampSchema


class FeedbackType(enum.Enum):
    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
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


####################
# SCHEMA
####################


class FeedbackBaseModel(BaseModel):
    type: Optional[FeedbackType] = FeedbackType.POSITIVE
    message: Optional[str] = None
    sender: Optional[str] = None


class FeedbackModel(TimestampSchema):
    pass


class FeedbackCreateModel(FeedbackBaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackReadModel(FeedbackBaseModel):
    id: UUID
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class FeedbackUpdate(FeedbackBaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackUpdateInternal(FeedbackUpdate):
    updated_at: datetime


CRUDFeedback = FastCRUD[
    Feedback,
    FeedbackBaseModel,
    FeedbackCreateModel,
    FeedbackReadModel,
    FeedbackUpdate,
    FeedbackUpdateInternal,
]

feedbacks = CRUDFeedback(Feedback)
