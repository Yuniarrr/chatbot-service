import logging
from typing import Optional

from app.models.users import RegisterForm
from app.core.database import session_manager
from app.models.users import (
    users,
    UserCreateInternalModel,
    UserReadWithPasswordModel,
    Role,
    UserReadModel,
    UserUpdateModel,
)
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class UserService:
    async def create_new_user(
        self, full_name: str, email: str, password: str, role: Role
    ) -> Optional[UserReadModel]:
        try:
            async with session_manager.session() as db:
                user = UserCreateInternalModel(
                    full_name=full_name,
                    email=email,
                    password=password,
                    role=role.value,
                )
                new_user = await users.create(db=db, object=user, commit=True)
                await db.refresh(new_user)
                return UserReadModel.model_validate(new_user)
        except Exception as e:
            log.error(f"Error creating new user: {e}")
            raise DatabaseException(str(e))

    async def get_user_by_id(self, user_id: str) -> Optional[UserReadModel]:
        try:
            async with session_manager.session() as db:
                user = await users.get(db=db, id=user_id)

                if not user:
                    return None

                return UserReadModel.model_validate(user)
        except Exception as e:
            log.error(f"Error get user by id: {e}")
            raise DatabaseException(str(e))

    async def get_user_by_email(
        self, email: str
    ) -> Optional[UserReadWithPasswordModel]:
        try:
            async with session_manager.session() as db:
                user = await users.get(
                    db=db,
                    email=email,
                    schema_to_select=UserReadWithPasswordModel,
                    return_as_model=True,
                )

                if not user:
                    return None

                return UserReadWithPasswordModel.model_validate(user)
        except Exception as e:
            log.error(f"Error get user by email: {e}")
            raise DatabaseException(str(e))

    async def get_users(self, skip: Optional[int] = None, limit: Optional[int] = None):
        try:
            async with session_manager.session() as db:
                return await users.get_multi(
                    db=db, offset=skip, limit=limit, schema_to_select=UserReadModel
                )
        except Exception as e:
            log.error(f"Error get users: {e}")
            raise DatabaseException(str(e))

    async def update_user_role_by_id(
        self, id: str, role: Role
    ) -> Optional[UserReadModel]:
        try:
            async with session_manager.session() as db:
                await users.update(
                    db=db,
                    object={"role": role.value},
                    id=id,
                )
        except Exception as e:
            log.error(f"Error updating user role: {e}")
            raise DatabaseException(str(e))

    async def update_user_by_id(
        self, id: str, data: UserUpdateModel
    ) -> Optional[UserReadModel]:
        try:
            async with session_manager.session() as db:
                await users.update(
                    db=db,
                    object=data.model_dump(),
                    id=id,
                )
        except Exception as e:
            log.error(f"Error updating user: {e}")
            raise DatabaseException(str(e))

    async def delete_user_by_id(self, id: str):
        try:
            async with session_manager.session() as db:
                await users.db_delete(db=db, id=id)
        except Exception as e:
            log.error(f"Error deleting user: {e}")
            raise DatabaseException(str(e))


user_service = UserService()
