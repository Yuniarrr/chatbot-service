import logging
from typing import Optional

from app.core.database import session_manager
from app.models.conversations import (
    ConversationReadModel,
    ConversationCreateModel,
    conversations,
    ConversationReadWithMessageModel,
    ConversationUpdateModel,
)
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class ConversationService:
    async def create_new_conversation(
        self, title: str, user_id: Optional[str] = None, sender: Optional[str] = None
    ) -> Optional[ConversationReadModel]:
        try:
            async with session_manager.session() as db:
                conversation = ConversationCreateModel(
                    title=title, user_id=user_id, sender=sender
                )
                new_conversation = await conversations.create(
                    db=db, object=conversation, commit=True
                )
                await db.refresh(new_conversation)
                return ConversationReadModel.model_validate(new_conversation)
        except Exception as e:
            raise DatabaseException(str(e))

    async def get_conversations_by_user_id(
        self, user_id: str, skip: Optional[int] = None, limit: Optional[int] = None
    ) -> Optional[ConversationReadModel]:
        try:
            async with session_manager.session() as db:
                return await conversations.get_multi(
                    db=db,
                    id=user_id,
                    offset=skip,
                    limit=limit,
                )
        except Exception as e:
            raise DatabaseException(str(e))

    async def get_conversation_with_message_by_id(
        self, id: str
    ) -> Optional[ConversationReadWithMessageModel]:
        try:
            async with session_manager.session() as db:
                return await conversations.get_joined(
                    db=db, id=id, join_schema_to_select=ConversationReadWithMessageModel
                )
        except Exception as e:
            raise DatabaseException(str(e))

    async def get_conversation_by_id(self, id: str) -> Optional[ConversationReadModel]:
        try:
            async with session_manager.session() as db:
                conversation = await conversations.get(db=db, id=id)

                if not conversation:
                    return None

                return ConversationReadModel.model_validate(conversation)
        except Exception as e:
            raise DatabaseException(str(e))

    async def get_conversation_by_sender(
        self, sender: str
    ) -> Optional[ConversationReadModel]:
        try:
            async with session_manager.session() as db:
                conversation = await conversations.get(db=db, sender=sender)

                if not conversation:
                    return None

                return ConversationReadModel.model_validate(conversation)
        except Exception as e:
            raise DatabaseException(str(e))

    async def update_user_by_id(
        self, id: str, data: ConversationUpdateModel
    ) -> Optional[ConversationReadModel]:
        try:
            async with session_manager.session() as db:
                return await conversations.update(
                    db=db,
                    object=data.model_dump(),
                    id=id,
                )
        except Exception as e:
            raise DatabaseException(str(e))

    async def delete_conversation_by_id(self, id: str):
        try:
            async with session_manager.session() as db:
                await conversations.db_delete(db=db, id=id, allow_multiple=False)
        except Exception as e:
            raise DatabaseException(str(e))


conversation_service = ConversationService()
