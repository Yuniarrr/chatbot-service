import enum
import uuid as uuid_pkg

from uuid import UUID
from typing import Union, Optional, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, Enum as SQLEnum, Text, ForeignKey, JSON
from fastcrud import FastCRUD

from app.core.database import Base
from app.core.schemas import TimestampSchema


class FileStatus(enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    AWAITING = "AWAITING"
    DETACHED = "DETACHED"


class File(Base):
    __tablename__ = "file"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )

    file_name: Mapped[Union[str, None]] = mapped_column(Text, index=True)
    file_path: Mapped[Union[str, None]] = mapped_column(Text, index=True)
    status: Mapped[FileStatus] = mapped_column(
        SQLEnum(FileStatus),
        default=FileStatus.FAILED,
        nullable=False,
    )
    meta: Mapped[str] = mapped_column(JSON, nullable=False)
    data: Mapped[str] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("UTC"))
    )
    updated_at: Mapped[Union[datetime, None]] = mapped_column(
        DateTime(timezone=True), default=None
    )

    user_id: Mapped[Union[str, None]] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )


####################
# FORMS
####################


class UpdateFileForm(BaseModel):
    status: Optional[FileStatus] = None
    meta: Optional[dict] = None


####################
# SCHEMA
####################


class FileBaseModel(BaseModel):
    id: UUID
    file_name: str
    file_path: str
    status: FileStatus
    meta: Optional[dict] = None
    data: Optional[dict] = None
    user_id: UUID

    class Config(ConfigDict):
        from_attributes = True


class FileModel(TimestampSchema):
    pass


class FileCreateModel(FileBaseModel):
    model_config = ConfigDict(extra="forbid")


class FileReadModel(FileBaseModel):
    created_at: Annotated[datetime, Field(examples=["datetime"])]
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        extra = "allow"


class FileUpdateModel(BaseModel):
    status: Optional[FileStatus] = None
    meta: Optional[dict] = None


class FileUpdateInternalModel(FileUpdateModel):
    updated_at: datetime


class ProcessFileForm(BaseModel):
    file_id: str
    content: Optional[str] = None
    collection_name: Optional[str] = None
    user_id: Optional[str] = None


CRUDFile = FastCRUD[
    FileBaseModel,
    FileModel,
    FileCreateModel,
    FileReadModel,
    FileUpdateModel,
    FileUpdateInternalModel,
]

files = CRUDFile(File)
