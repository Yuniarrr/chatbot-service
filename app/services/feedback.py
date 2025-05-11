import logging

from app.models.feedbacks import FeedbackCreateModel, FeedbackReadModel, feedbacks
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


feedback_service = FeedbackService()
