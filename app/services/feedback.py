import logging

from typing import Optional
from sqlalchemy import func, or_
from sqlalchemy.future import select

from app.models.feedbacks import (
    FeedbackCreateModel,
    FeedbackReadModel,
    feedbacks,
    Feedback,
)
from app.core.database import session_manager
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class FeedbackService:
    async def insert_new_feedback(
        self, form_data: FeedbackCreateModel
    ) -> FeedbackReadModel:
        try:
            async with session_manager.session() as db:
                new_feedback = await feedbacks.create(
                    db=db, object=form_data, commit=True
                )
                await db.refresh(new_feedback)
                return FeedbackReadModel.model_validate(new_feedback)
        except Exception as e:
            log.error(f"Error inserting new feedback: {e}")
            raise DatabaseException(str(e))

    async def get_feedbacks(
        self,
        skip: Optional[int] = 0,
        limit: Optional[int] = 10,
        search: Optional[str] = None,
    ):
        try:
            async with session_manager.session() as db:
                stmt = select(Feedback)

                if search:
                    stmt = stmt.where(
                        or_(
                            Feedback.message.ilike(f"%{search}%"),
                            Feedback.sender.ilike(f"%{search}%"),
                        )
                    )

                count_stmt = select(func.count()).select_from(stmt.subquery())
                total_result = await db.execute(count_stmt)
                total = total_result.scalar() or 0

                stmt = (
                    stmt.offset(skip).limit(limit).order_by(Feedback.created_at.desc())
                )

                result = await db.execute(stmt)
                feedbacks_list = result.scalars().all()

                return {
                    "data": [
                        FeedbackReadModel.model_validate(f) for f in feedbacks_list
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
            log.error(f"Error get feedbacks: {e}")
            raise DatabaseException(str(e))

    async def get_feedback_by_id(self, id: str) -> Optional[FeedbackReadModel]:
        try:
            async with session_manager.session() as db:
                feedback = await feedbacks.get(db=db, id=id)
                if not feedback:
                    return None
                return FeedbackReadModel.model_validate(feedback)
        except Exception as e:
            log.error(f"Error get feedback by id: {e}")
            raise DatabaseException(str(e))

    async def delete_feedback_by_id(self, id: str):
        try:
            async with session_manager.session() as db:
                await feedbacks.db_delete(db=db, id=id, allow_multiple=False)
        except Exception as e:
            log.error(f"Error deleting feedback: {e}")
            raise DatabaseException(str(e))


feedback_service = FeedbackService()
