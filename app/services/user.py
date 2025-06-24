import logging
from typing import Optional

from sqlalchemy import func, or_
from sqlalchemy.future import select

from app.models.users import RegisterForm
from app.core.database import session_manager
from app.models.users import (
    users,
    UserCreateInternalModel,
    UserReadWithPasswordModel,
    Role,
    UserReadModel,
    UserUpdateModel,
    User,
)
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class UserService:
    async def create_new_user(
        self, full_name: str, email: str, password: str, role: Role, phone_number: Optional[str] = None
    ) -> Optional[UserReadModel]:
        try:
            async with session_manager.session() as db:
                user = UserCreateInternalModel(
                    full_name=full_name,
                    email=email,
                    password=password,
                    role=role.value,
                    phone_number=phone_number,
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

    async def get_user_by_user_id(
        self, user_id: str
    ) -> Optional[UserReadWithPasswordModel]:
        try:
            async with session_manager.session() as db:
                user = await users.get(
                    db=db,
                    user_id=user_id,
                    schema_to_select=UserReadWithPasswordModel,
                    return_as_model=True,
                )

                if not user:
                    return None

                return UserReadWithPasswordModel.model_validate(user)
        except Exception as e:
            log.error(f"Error get user by email: {e}")
            raise DatabaseException(str(e))

    async def get_users(
        self,
        skip: Optional[int] = 0,
        limit: Optional[int] = 10,
        search: Optional[str] = None,
    ):
        try:
            async with session_manager.session() as db:
                stmt = select(User)

                if search:
                    stmt = stmt.where(
                        or_(
                            User.full_name.ilike(f"%{search}%"),
                            User.email.ilike(f"%{search}%"),
                        )
                    )

                # Hitung total dulu (setelah filter)
                count_stmt = select(func.count()).select_from(stmt.subquery())
                total_result = await db.execute(count_stmt)
                total = total_result.scalar() or 0

                # Tambahkan offset & limit untuk paginasi
                stmt = stmt.offset(skip).limit(limit)

                result = await db.execute(stmt)
                users_list = result.scalars().all()

                return {
                    "data": [UserReadModel.model_validate(u) for u in users_list],
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
        self, id: str, form_data: UserUpdateModel
    ) -> Optional[UserReadModel]:
        try:
            async with session_manager.session() as db:
                updated_user = await users.update(
                    db=db,
                    object=form_data,
                    id=id,
                    commit=True,
                    return_as_model=True,
                    schema_to_select=User,
                )
                updated_user = await db.merge(updated_user)
                await db.refresh(updated_user)
                return UserReadModel.model_validate(updated_user)
        except Exception as e:
            log.error(f"Error updating user: {e}")
            raise DatabaseException(str(e))

    async def delete_user_by_id(self, id: str):
        try:
            async with session_manager.session() as db:
                await users.db_delete(db=db, id=id, allow_multiple=False)
        except Exception as e:
            log.error(f"Error deleting user: {e}")
            raise DatabaseException(str(e))


user_service = UserService()
