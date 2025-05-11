import logging
from typing import Optional

from app.core.database import session_manager
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException
from app.models.opportunities import (
    OpportunitiesCreateModel,
    OpportunitiesReadModel,
    OpportunityType,
    opportunities,
    OpportunitiesUpdateModel,
    Opportunity,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class OpportunityService:
    async def insert_new_opportunity(
        self, form_data: OpportunitiesCreateModel, uploader: str
    ) -> OpportunitiesReadModel:
        try:
            data = form_data.model_dump()
            data["uploader"] = uploader

            new_opportunity_data = Opportunity(**data)
            async with session_manager.session() as db:
                new_opportunity = await opportunities.create(
                    db=db, object=new_opportunity_data, commit=True
                )
                await db.refresh(new_opportunity)
                return OpportunitiesReadModel.model_validate(new_opportunity)
        except Exception as e:
            log.error(f"Error inserting new opportunity: {e}")
            raise DatabaseException(str(e))

    async def update_opportunity_by_id(
        self, opportunity_id: str, form_data: OpportunitiesUpdateModel
    ) -> OpportunitiesReadModel:
        try:
            async with session_manager.session() as db:
                updated_opportunity = await opportunities.update(
                    db=db,
                    object=form_data,
                    commit=True,
                    return_as_model=True,
                    schema_to_select=Opportunity,
                    id=opportunity_id,
                )
                updated_opportunity = await db.merge(updated_opportunity)
                await db.refresh(updated_opportunity)
                return OpportunitiesReadModel.model_validate(updated_opportunity)
        except Exception as e:
            log.error(f"Error updating opportunity: {e}")
            raise DatabaseException(str(e))

    async def delete_opportunity_by_id(self, id: str):
        try:
            async with session_manager.session() as db:
                await opportunities.db_delete(db, id)
        except Exception as e:
            log.error(f"Error deleting opportunity: {e}")
            raise DatabaseException(str(e))

    async def get_opportunities(
        self,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
        type: Optional[OpportunityType] = None,
    ):
        try:
            async with session_manager.session() as db:
                opportunity_list = await opportunities.get_multi(
                    db=db, skip=skip, limit=limit, type=type
                )
                return opportunity_list
        except Exception as e:
            log.error(f"Error get opportunities: {e}")
            raise DatabaseException(str(e))

    async def get_opportunity_by_id(self, id: str) -> Optional[OpportunitiesReadModel]:
        try:
            async with session_manager.session() as db:
                opportunity = await opportunities.get(db=db, id=id)
                if not opportunity:
                    return None
                return OpportunitiesReadModel.model_validate(opportunity)
        except Exception as e:
            log.error(f"Error get opportunity by id: {e}")
            raise DatabaseException(str(e))


opportunity_service = OpportunityService()
