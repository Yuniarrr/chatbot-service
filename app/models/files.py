import uuid as uuid_pkg

from uuid import UUID
from enum import Enum
from typing import Union, Optional, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, String, Enum as SQLEnum, String, Text, ForeignKey
from fastcrud import FastCRUD

from app.core.database import Base
from app.core.schemas import TimestampSchema


class FileStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    AWAITING = "AWAITING"


class File(Base):
    __tablename__ = "file"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )

    file_name: Mapped[Union[str, None]] = mapped_column(String(40), index=True)
    file_path: Mapped[Union[str, None]] = mapped_column(Text, index=True)
    status: Mapped[FileStatus] = mapped_column(
        SQLEnum(FileStatus), default=FileStatus.FAILED, nullable=False
    )

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

    uploader = relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="uploader_file",
        lazy="selectin",
    )


####################
# SCHEMA
####################


class FileBaseModel(BaseModel):
    file_name: str
    file_path: str
    status: FileStatus
    user_id: str

    class Config(ConfigDict):
        orm_mode = True


class FileModel(TimestampSchema):
    pass


class FileCreateModel(FileBaseModel):
    model_config = ConfigDict(extra="forbid")


class FileReadModel(FileBaseModel):
    id: UUID

    created_at: Annotated[datetime, Field(examples=["datetime"])]
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class FileUpdateModel(BaseModel):
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    status: Optional[FileStatus] = None


class FileUpdateInternalModel(FileUpdateModel):
    updated_at: datetime


CRUDFile = FastCRUD[
    FileBaseModel,
    FileModel,
    FileCreateModel,
    FileReadModel,
    FileUpdateModel,
    FileUpdateInternalModel,
]

files = CRUDFile(File)
