import logging
from typing import Optional

from app.core.database import session_manager
from app.models.messages import MessageReadModel, MessageCreateModel, messages
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class MessageService:
    async def create_new_message(
        self,
        form_data: MessageCreateModel,
    ) -> Optional[MessageReadModel]:
        try:
            async with session_manager.session() as db:
                new_message = await messages.create(
                    db=db, object=form_data, commit=True
                )
                await db.refresh(new_message)
                return MessageReadModel.model_validate(new_message)
        except Exception as e:
            log.error(f"Error create_new_message: {e}")
            raise DatabaseException(str(e))

    async def get_messages_by_conversation_id(
        self,
        conversation_id: str,
        skip: Optional[int] = 0,
        limit: Optional[int] = 10,
    ):
        try:
            async with session_manager.session() as db:
                return await messages.get_multi(
                    db=db,
                    conversation_id=conversation_id,
                    offset=skip,
                    limit=limit,
                    sort_columns="created_at",
                    sort_orders="desc",
                )
        except Exception as e:
            log.error(f"Error get_messages_by_conversation_id: {e}")
            raise DatabaseException(str(e))


message_service = MessageService()
