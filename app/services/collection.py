from datetime import datetime
import logging

from typing import Optional
from zoneinfo import ZoneInfo
from sqlalchemy.future import select

from app.core.constants import ERROR_MESSAGES
from app.core.logger import SRC_LOG_LEVELS
from app.models.collections import collections
from app.models.collections import (
    CollectionCreateModel,
    CollectionReadModel,
    Collection,
    CollectionUpdateModel,
)
from app.core.database import session_manager
from app.core.exceptions import DatabaseException, NotFoundException

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class CollectionService:
    async def insert_new_collection(
        self, form_data: CollectionCreateModel
    ) -> CollectionReadModel:
        try:
            async with session_manager.session() as db:
                new_collection = await collections.create(
                    db=db, object=form_data, commit=True
                )
                await db.refresh(new_collection)
                return CollectionReadModel.model_validate(new_collection)
        except Exception as e:
            log.error(f"Error inserting new collection: {e}")
            raise DatabaseException(str(e))

    async def get_active_collections(
        self,
    ):
        try:
            async with session_manager.session() as db:
                collections_list = await collections.get_multi(db=db, is_active=True)
                if not collections_list:
                    return []
                return collections_list
        except Exception as e:
            log.error(f"Error get active collections: {e}")
            raise DatabaseException(str(e))

    async def get_collections(
        self,
    ):
        try:
            async with session_manager.session() as db:
                collections_list = await collections.get_multi(db=db)
                if not collections_list:
                    return []
                return collections_list
        except Exception as e:
            log.error(f"Error get collections: {e}")
            raise DatabaseException(str(e))

    async def update_count_collection_by_collection_name(
        self, collection_name: str, is_add: bool = True
    ) -> CollectionReadModel | None:
        try:
            async with session_manager.session() as db:
                result = await db.execute(
                    select(Collection).filter(Collection.name == collection_name)
                )
                collection = result.scalar_one_or_none()

                if not collection:
                    raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("jenis dokumen"))

                if is_add:
                    collection.count = (collection.count or 0) + 1
                else:
                    collection.count = (collection.count or 0) - 1
                collection.updated_at = datetime.now(ZoneInfo("UTC"))

                await db.commit()
                await db.refresh(collection)

                return CollectionReadModel.model_validate(collection)
        except Exception as e:
            log.error(f"Error update_count_collection_by_collection_name count: {e}")
            raise DatabaseException(str(e))

    async def find_collection_by_name(
        self, collection_name: str
    ) -> CollectionReadModel | None:
        try:
            async with session_manager.session() as db:
                collection = await collections.get(
                    db=db,
                    collection_name=collection_name,
                    schema_to_select=CollectionReadModel,
                )
                if not collection:
                    return None
                return CollectionReadModel.model_validate(collection)
        except Exception as e:
            log.error(f"Error find_collection_by_name count: {e}")
            raise DatabaseException(str(e))

    async def get_collection_by_id(
        self, collection_id: str
    ) -> CollectionReadModel | None:
        try:
            async with session_manager.session() as db:
                collection = await collections.get(
                    db=db,
                    id=collection_id,
                    schema_to_select=CollectionReadModel,
                )
                if not collection:
                    return None
                return CollectionReadModel.model_validate(collection)
        except Exception as e:
            log.error(f"Error get_collection_by_id: {e}")
            raise DatabaseException(str(e))

    async def update_status_by_collection_id(
        self, collection_id: str, status: bool
    ) -> CollectionReadModel | None:
        try:
            async with session_manager.session() as db:
                updated_col = await collections.update(
                    db=db,
                    object={"is_active": status},
                    id=collection_id,
                    commit=True,
                    return_as_model=True,
                    schema_to_select=Collection,
                )
                updated_col = await db.merge(updated_col)
                await db.refresh(updated_col)
                return CollectionReadModel.model_validate(updated_col)
        except Exception as e:
            log.error(f"Error update_status_by_collection_id: {e}")
            raise DatabaseException(str(e))


collection_service = CollectionService()
