import logging
from typing import Optional

from app.core.database import async_get_db
from app.models.messages import MessageReadModel, MessageCreateModel, messages
from app.core.logger import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class MessaeService:
    async def create_new_message(
        self,
        data: MessageCreateModel,
    ) -> Optional[MessageReadModel]:
        try:
            async with async_get_db() as db:
                new_message = await messages.create(db=db, object=data.model_dump())
                return MessageReadModel.model_validate(new_message)
        except Exception:
            return None

    async def get_messages_by_conversation_id(
        self,
        conversation_id: str,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        try:
            async with async_get_db() as db:
                return await messages.get_multi(
                    db=db,
                    conversation_id=conversation_id,
                    offset=skip,
                    limit=limit,
                    sort_columns="created_at",
                    sort_orders="desc",
                )
        except Exception:
            return None
