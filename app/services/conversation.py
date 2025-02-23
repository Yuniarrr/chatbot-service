import logging
from typing import Optional

from app.core.database import async_get_db
from app.models.conversations import (
    ConversationReadModel,
    ConversationCreateModel,
    conversations,
    ConversationReadWithMessageModel,
    ConversationUpdateModel,
)

log = logging.getLogger(__name__)
log.setLevel("SERVICE")


class ConversationService:
    async def create_new_conversation(
        self, title: str, user_id: str
    ) -> Optional[ConversationReadModel]:
        try:
            async with async_get_db() as db:
                conversation = ConversationCreateModel(
                    title=title,
                    user_id=user_id,
                )
                new_conversation = await conversations.create(
                    db=db, object=conversation
                )

                return ConversationReadModel.model_validate(new_conversation)
        except Exception:
            return None

    async def get_conversations_by_user_id(
        self, user_id: str, skip: Optional[int] = None, limit: Optional[int] = None
    ) -> Optional[ConversationReadModel]:
        try:
            async with async_get_db() as db:
                return await conversations.get_multi(
                    db=db,
                    id=user_id,
                    offset=skip,
                    limit=limit,
                )
        except Exception:
            return None

    async def get_conversation_by_id(
        self, id: str
    ) -> Optional[ConversationReadWithMessageModel]:
        try:
            async with async_get_db() as db:
                return await conversations.get_joined(
                    db=db, id=id, join_schema_to_select=ConversationReadWithMessageModel
                )
        except Exception:
            return None

    async def update_user_by_id(
        self, id: str, data: ConversationUpdateModel
    ) -> Optional[ConversationReadModel]:
        try:
            async with async_get_db() as db:
                return await conversations.update(
                    db=db,
                    object=data.model_dump(),
                    id=id,
                )
        except Exception:
            return None

    async def delete_conversation_by_id(self, id: str):
        try:
            async with async_get_db() as db:
                await conversations.db_delete(db=db, id=id)
        except Exception:
            return None


conversation_service = ConversationService()
