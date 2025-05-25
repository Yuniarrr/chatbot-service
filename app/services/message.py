import logging

from typing import Optional
from sqlalchemy.future import select

from app.core.database import session_manager
from app.models.messages import MessageReadModel, MessageCreateModel, messages, Message
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
                stmt = (
                    select(Message)
                    .filter(Message.conversation_id == conversation_id)
                    .offset(skip)
                    .limit(limit)
                    .order_by(Message.created_at.desc())
                )
                result = await db.execute(stmt)
                messages_list = result.scalars().all()
                total = await messages.count(db=db, conversation_id=conversation_id)

                if not messages_list:
                    return {
                        "data": [],
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

                return {
                    "data": [MessageReadModel.model_validate(m) for m in messages_list],
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
            log.error(f"Error get_messages_by_conversation_id: {e}")
            raise DatabaseException(str(e))


message_service = MessageService()
