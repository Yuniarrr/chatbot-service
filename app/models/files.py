import uuid as uuid_pkg

from enum import Enum
from typing import Union
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, String, Enum as SQLEnum, String, Text, ForeignKey

from app.core.database import Base


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

    uploaded_by: Mapped[Union[str, None]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )

    uploader = relationship(
        "User",
        foreign_keys=[uploaded_by],
        back_populates="uploader_file",
        lazy="selectin",
    )
