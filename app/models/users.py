import logging
import uuid as uuid_pkg

from enum import Enum
from typing import Union, List, Optional, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, String, Enum as SQLEnum, String, relationship, Text
from fastcrud import FastCRUD

from app.core.database import Base
from app.models.conversations import Conversation
from app.core.schemas import TimestampSchema

log = logging.getLogger(__name__)
log.setLevel("MODELS")


class Role(Enum):
    ADMINISTRATOR = "ADMINISTRATOR"
    DOSEN = "DOSEN"
    MAHASISWA = "MAHASISWA"


class User(Base):
    __tablename__ = "user"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )

    username: Mapped[Union[str, None]] = mapped_column(String(40))
    email: Mapped[str] = mapped_column(String(40), unique=True, unique=True)
    phone_number: Mapped[Union[str, None]] = mapped_column(
        String(20), unique=True, nullable=True
    )
    password: Mapped[Union[str, None]] = mapped_column(String(255), nullable=True)
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


####################
# FORMS
####################


class SigninForm(BaseModel):
    email: str
    password: str


class SignupForm(BaseModel):
    username: str
    email: str
    password: str


####################
# SCHEMA
####################


class UserBaseModel(BaseModel):
    username: str
    email: Optional[str] = None
    phone_number: Optional[str] = None
    # password: Optional[str] = None
    role: Role
    profile_picture: Optional[str] = None
    nrp: Optional[str] = None
    nip: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class UserModel(TimestampSchema):
    pass


class UserReadModel(UserBaseModel):
    class Config:
        from_attributes = True


class UserCreateInternalModel(UserBaseModel):
    password: Annotated[
        str,
        Field(
            # pattern=r"^.{8,}|[0-9]+|[A-Z]+|[a-z]+|[^a-zA-Z0-9]+$",
            examples=["string"]
        ),
    ]


class UserUpdateModel(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    password: Optional[str] = None
    role: Optional[Role] = None
    profile_picture: Optional[str] = None
    nrp: Optional[str] = None
    nip: Optional[str] = None


class UserUpdateInternalModel(UserUpdateModel):
    updated_at: datetime


CRUDUser = FastCRUD[
    User, UserCreateInternalModel, UserUpdateModel, UserUpdateInternalModel
]

crud_user = CRUDUser(User)
