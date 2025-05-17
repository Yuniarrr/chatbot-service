import logging
from typing import Optional

from app.core.logger import SRC_LOG_LEVELS
from app.models.collections import collections
from app.models.collections import CollectionCreateModel, CollectionReadModel
from app.core.database import session_manager
from app.core.exceptions import DatabaseException

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


collection_service = CollectionService()
