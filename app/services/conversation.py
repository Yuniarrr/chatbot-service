import logging
from typing import Optional

from sqlalchemy.future import select
from sqlalchemy.sql import or_, join, desc
from datetime import datetime, time, timezone, timedelta

from app.core.database import session_manager
from app.models.conversations import (
    Conversation,
    ConversationReadModel,
    ConversationCreateModel,
    conversations,
    ConversationReadWithMessageModel,
    ConversationUpdateModel,
)
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException
from app.models.users import User

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

    async def get_unique_conversation_items(
        self, skip: int = 0, limit: int = 100
    ) -> dict[str, object]:
        try:
            async with session_manager.session() as db:
                query = select(
                    Conversation.sender,
                    Conversation.user_id,
                    User.full_name,
                ).outerjoin(User, User.id == Conversation.user_id)

                result = await db.execute(query)
                rows: list[tuple[str | None, str | None, str | None]] = result.all()

                seen = set()
                unique_items = []

                for sender, user_id, full_name in rows:
                    key = (user_id or "", sender or "")
                    if key not in seen:
                        seen.add(key)
                        unique_items.append(
                            {
                                "user_id": user_id,
                                "full_name": full_name,
                                "sender": sender,
                            }
                        )

                total = len(unique_items)
                paginated_items = unique_items[skip : skip + limit]

                return {
                    "data": paginated_items,
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
            raise DatabaseException(str(e))

    async def get_conversations_by_user_id(
        self, user_id: str, skip: Optional[int] = None, limit: Optional[int] = None
    ):
        try:
            async with session_manager.session() as db:
                stmt = (
                    select(Conversation)
                    .offset(skip)
                    .limit(limit)
                    .filter(Conversation.user_id == user_id)
                    .order_by(Conversation.created_at.desc())
                )
                result = await db.execute(stmt)
                conversations_list = result.scalars().all()

                total = await conversations.count(db=db, user_id=user_id)

                if not conversations_list:
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
                    "data": [
                        ConversationReadModel.model_validate(c)
                        for c in conversations_list
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
        self, sender: str, skip: Optional[int] = 0, limit: Optional[int] = 10
    ):
        try:
            async with session_manager.session() as db:
                stmt = (
                    select(Conversation)
                    .offset(skip)
                    .limit(limit)
                    .filter(Conversation.sender == sender)
                    .order_by(Conversation.created_at.desc())
                )
                result = await db.execute(stmt)
                conversations_list = result.scalars().all()
                total = await conversations.count(db=db, sender=sender)

                if not conversations_list:
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
                    "data": [
                        ConversationReadModel.model_validate(c)
                        for c in conversations_list
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

    async def get_one_conversation_by_sender(self, sender: str):
        try:
            async with session_manager.session() as db:
                query = (
                    select(Conversation)
                    .where(Conversation.sender == sender)
                    .order_by(desc(Conversation.created_at))
                    .limit(1)
                )

                result = await db.execute(query)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    return None

                return ConversationReadModel.model_validate(conversation)
        except Exception as e:
            raise DatabaseException(str(e))

    async def delete_conversation_by_id(self, id: str):
        try:
            async with session_manager.session() as db:
                await conversations.db_delete(db=db, id=id, allow_multiple=False)
        except Exception as e:
            raise DatabaseException(str(e))

    async def delete_conversation_by_sender(self, sender: str):
        try:
            async with session_manager.session() as db:
                await conversations.db_delete(db=db, sender=sender, allow_multiple=True)
        except Exception as e:
            raise DatabaseException(str(e))

    async def delete_conversation_by_user_id(self, user_id: str):
        try:
            async with session_manager.session() as db:
                await conversations.db_delete(
                    db=db, user_id=user_id, allow_multiple=True
                )
        except Exception as e:
            raise DatabaseException(str(e))

    async def get_today_conversation_by_sender(self, sender: str):
        try:
            async with session_manager.session() as db:
                today_start = datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

                query = (
                    select(Conversation)
                    .where(
                        Conversation.sender == sender,
                        Conversation.created_at >= today_start,
                    )
                    .order_by(desc(Conversation.created_at))
                    .limit(1)
                )

                result = await db.execute(query)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    return None

                return ConversationReadModel.model_validate(conversation)
        except Exception as e:
            raise DatabaseException(str(e))


conversation_service = ConversationService()
