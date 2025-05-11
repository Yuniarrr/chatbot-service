import enum
from typing import Optional, Union
import uuid as uuid_pkg

from uuid import UUID
from sqlalchemy.orm import Mapped, mapped_column
from zoneinfo import ZoneInfo
from sqlalchemy import DateTime, String, Enum as SQLEnum, String, Text, Date
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from fastcrud import FastCRUD

from app.core.database import Base
from app.core.schemas import TimestampSchema


class OpportunityType(enum.Enum):
    BEASISWA = "BEASISWA"
    MAGANG = "MAGANG"
    LOMBA = "LOMBA"
    SERTIFIKASI = "SERTIFIKASI"


class Opportunity(Base):
    __tablename__ = "opportunity"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        "id", default=uuid_pkg.uuid4, primary_key=True, unique=True
    )

    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Union[str, None]] = mapped_column(Text, nullable=True)
    organizer: Mapped[Union[str, None]] = mapped_column(String(50), nullable=True)
    type: Mapped[OpportunityType] = mapped_column(
        SQLEnum(OpportunityType),
        default=OpportunityType.SERTIFIKASI,
        nullable=False,
    )
    start_date: Mapped[Union[Date, None]] = mapped_column(Date, nullable=True)
    end_date: Mapped[Union[Date, None]] = mapped_column(Date, nullable=True)
    link: Mapped[Union[str, None]] = mapped_column(Text, nullable=True)
    image_url: Mapped[Union[str, None]] = mapped_column(Text, nullable=True)

    uploader: Mapped[Union[str, None]] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(ZoneInfo("UTC"))
    )
    updated_at: Mapped[Union[datetime, None]] = mapped_column(
        DateTime(timezone=True), default=None
    )


####################
# FORMS
####################


class OpportunitiesForm(BaseModel):
    title: str
    description: Optional[str] = None
    organizer: Optional[str] = None
    type: OpportunityType = OpportunityType.SERTIFIKASI
    start_date: Optional[Date] = None
    end_date: Optional[Date] = None
    link: Optional[str] = None
    image_url: Optional[str] = None


####################
# SCHEMA
####################


class OpportunitiesBaseModel(BaseModel):
    title: str
    description: Optional[str] = None
    organizer: Optional[str] = None
    type: OpportunityType = OpportunityType.SERTIFIKASI
    start_date: Optional[Date] = None
    end_date: Optional[Date] = None
    link: Optional[str] = None
    image_url: Optional[str] = None
    uploader: Optional[str] = None


class OpportunitiesModel(TimestampSchema):
    pass


class OpportunitiesCreateModel(OpportunitiesBaseModel):
    model_config = ConfigDict(extra="forbid")


class OpportunitiesReadModel(OpportunitiesBaseModel):
    id: UUID
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class OpportunitiesUpdateModel(OpportunitiesBaseModel):
    model_config = ConfigDict(extra="forbid")


class OpportunitiesUpdateInternalModel(OpportunitiesUpdateModel):
    updated_at: datetime


CRUDOpportunities = FastCRUD[
    Opportunity,
    OpportunitiesBaseModel,
    OpportunitiesCreateModel,
    OpportunitiesReadModel,
    OpportunitiesUpdateModel,
    OpportunitiesUpdateInternalModel,
]

opportunities = CRUDOpportunities(Opportunity)
