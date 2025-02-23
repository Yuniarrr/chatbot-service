import logging
from typing import Optional

from app.models.users import RegisterForm
from app.core.database import async_get_db
from app.models.users import (
    users,
    UserCreateInternalModel,
    Role,
    UserReadModel,
    UserUpdateModel,
)

log = logging.getLogger(__name__)
log.setLevel("SERVICE")


class UserService:
    async def create_new_user(
        self, full_name: str, email: str, password: str, role: Role
    ) -> Optional[UserReadModel]:
        try:
            async with async_get_db() as db:
                user = UserCreateInternalModel(
                    full_name=full_name,
                    email=email,
                    password=password,
                    role=role.value,
                )
                new_user = await users.create(db=db, object=user)

                return UserReadModel.model_validate(new_user)
        except Exception:
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[UserReadModel]:
        try:
            async with async_get_db() as db:
                user = await users.get(db=db, id=user_id)
                return UserReadModel.model_validate(user)
        except Exception:
            return None

    async def get_user_by_email(self, email: str) -> Optional[UserReadModel]:
        try:
            async with async_get_db() as db:
                user = await users.get(db=db, email=email)
                return UserReadModel.model_validate(user)
        except Exception:
            return None

    async def get_users(self, skip: Optional[int] = None, limit: Optional[int] = None):
        try:
            async with async_get_db() as db:
                return await users.get_multi(db=db, offset=skip, limit=limit)
        except Exception:
            return None

    async def update_user_role_by_id(
        self, id: str, role: Role
    ) -> Optional[UserReadModel]:
        try:
            async with async_get_db() as db:
                await users.update(
                    db=db,
                    object={"role": role.value},
                    id=id,
                )
        except Exception:
            return None

    async def update_user_by_id(
        self, id: str, data: UserUpdateModel
    ) -> Optional[UserReadModel]:
        try:
            async with async_get_db() as db:
                await users.update(
                    db=db,
                    object=data.model_dump(),
                    id=id,
                )
        except Exception:
            return None

    async def delete_user_by_id(self, id: str):
        try:
            async with async_get_db() as db:
                await users.db_delete(db=db, id=id)
        except Exception:
            return None


user_service = UserService()
