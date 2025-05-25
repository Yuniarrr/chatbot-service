import logging

from typing import Optional
from datetime import datetime
from sqlalchemy import func, or_
from sqlalchemy.future import select

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
        self, form_data: OpportunitiesCreateModel
    ) -> OpportunitiesReadModel:
        try:
            async with session_manager.session() as db:
                new_opportunity = await opportunities.create(
                    db=db, object=form_data, commit=True
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
                await opportunities.db_delete(db=db, id=id, allow_multiple=False)
        except Exception as e:
            log.error(f"Error deleting opportunity: {e}")
            raise DatabaseException(str(e))

    async def get_opportunities(
        self,
        skip: Optional[int] = 0,
        limit: Optional[int] = 10,
        type: Optional[OpportunityType] = None,
        search: Optional[str] = None,
    ):
        try:
            async with session_manager.session() as db:
                stmt = select(Opportunity)

                if type:
                    stmt = stmt.where(Opportunity.type == type)

                if search:
                    stmt = stmt.where(
                        or_(
                            Opportunity.title.ilike(f"%{search}%"),
                            Opportunity.description.ilike(f"%{search}%"),
                            Opportunity.organizer.ilike(f"%{search}%"),
                        )
                    )

                # Hitung total setelah filter
                count_stmt = select(func.count()).select_from(stmt.subquery())
                total_result = await db.execute(count_stmt)
                total = total_result.scalar() or 0

                stmt = (
                    stmt.offset(skip)
                    .limit(limit)
                    .order_by(Opportunity.created_at.desc())
                )

                result = await db.execute(stmt)
                opportunities_list = result.scalars().all()

                return {
                    "data": [
                        OpportunitiesReadModel.model_validate(opp)
                        for opp in opportunities_list
                    ],
                    "meta": {
                        "skip": skip,
                        "limit": limit,
                        "total": total,
                        "is_next": skip + limit < total,
                        "is_prev": skip > 0,
                        "start": skip + 1 if total > 0 else 0,
                        "end": min(skip + limit, total),
                    },
                }
        except Exception as e:
            log.error(f"Error get opportunities: {e}")
            raise DatabaseException(str(e))

    async def get_opportunity_by_filter(
        self,
        type: Optional[OpportunityType] = None,
        search: Optional[str] = None,
        skip: Optional[int] = 0,
        limit: Optional[int] = 10,
    ) -> list[OpportunitiesReadModel]:
        try:
            async with session_manager.session() as db:
                # Menggunakan select() untuk membuat query asinkron
                query = select(Opportunity)

                # Filter by type if provided
                if type:
                    query = query.filter(Opportunity.type == type)

                # Filter by title if provided
                if search:
                    search_term = f"%{search}%"
                    query = query.filter(
                        or_(
                            Opportunity.title.ilike(search_term),
                            Opportunity.description.ilike(search_term),
                            Opportunity.organizer.ilike(search_term),
                            Opportunity.link.ilike(search_term),
                        )
                    )

                # Filter by end_date if it is provided and check if the date is passed
                query = query.filter(
                    or_(
                        Opportunity.end_date.is_(
                            None
                        ),  # If end_date is NULL, include it
                        Opportunity.end_date
                        > datetime.now(),  # If end_date is not passed, check if it is in the future
                    )
                )

                # Paginate the results
                query = query.offset(skip).limit(limit)

                # Execute the query
                result = await db.execute(query)

                # Fetch results
                opportunities = result.scalars().all()

                # Map results to schema models
                return [
                    OpportunitiesReadModel.model_validate(opportunity)
                    for opportunity in opportunities
                ]

        except Exception as e:
            log.error(f"Error getting opportunities: {e}")
            raise DatabaseException("Error getting opportunities from database")

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
