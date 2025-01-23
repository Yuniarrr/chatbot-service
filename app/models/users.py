import uuid as uuid_pkg

from enum import Enum
from typing import Union, List
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, String, Enum as SQLEnum, String, relationship, Text

from app.core.database import Base
from app.models.conversations import Conversation


class Role(Enum):
    ADMINISTRATOR = "ADMINISTRATOR"
    DOSEN = "DOSEN"
    MAHASISWA = "MAHASISWA"


class User(Base):
    __tablename__ = "user"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )

    username: Mapped[Union[str, None]] = mapped_column(String(40), index=True)
    email: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    phone_number: Mapped[Union[str, None]] = mapped_column(
        String(20), unique=True, index=True, nullable=True
    )
    role: Mapped[Role] = mapped_column(SQLEnum(Role), default=Role.USER, nullable=False)

    profile_picture: Mapped[Union[str, None]] = mapped_column(Text, nullable=True)
    nrp: Mapped[Union[str, None]] = mapped_column(String, nullable=True)
    nip: Mapped[Union[str, None]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("UTC"))
    )
    updated_at: Mapped[Union[datetime, None]] = mapped_column(
        DateTime(timezone=True), default=None
    )

    uploader_file = relationship(
        "File",
        foreign_keys="File.uploaded_by",
        back_populates="uploader",
        lazy="selectin",
    )

    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation",
        back_populates="user",
        lazy="selectin",
        cascade="all, delete",
        passive_deletes=True,
    )
