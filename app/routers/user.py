import logging

from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Query
from uuid import UUID

from app.core.response import ResponseModel
from app.models.users import (
    AddUserForm,
    Role,
    LoginForm,
    UpdateUserForm,
    UserUpdateModel,
)
from app.services.user import user_service
from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.utils.auth import (
    get_password_hash,
    TokenData,
    get_admin_user,
    get_not_user,
    get_verified_user,
)
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import (
    NotFoundException,
    DuplicateValueException,
    InternalServerException,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/", response_model=ResponseModel)
async def register(
    form_data: AddUserForm,
    current_user: Annotated[TokenData, Depends(get_admin_user)],
):
    try:
        is_email_exist = await user_service.get_user_by_email(form_data.email)

        if is_email_exist is not None:
            raise DuplicateValueException(ERROR_MESSAGES.DUPLICATE_VALUE("Email"))

        hash_password = get_password_hash(form_data.password)

        new_user = await user_service.create_new_user(
            full_name=form_data.full_name,
            email=form_data.email,
            password=hash_password,
            role=form_data.role,
            phone_number=form_data.phone_number,
        )

        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.CREATED, data=new_user
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.patch("/{user_id}", response_model=ResponseModel)
async def update_user(
    user_id: UUID,
    current_user: Annotated[TokenData, Depends(get_admin_user)],
    form_data: UpdateUserForm,
):
    try:
        is_user_exist = await user_service.get_user_by_user_id(str(user_id))

        if is_user_exist is None:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("Email"))

        if form_data.email:
            user_with_email = await user_service.get_user_by_email(form_data.email)
            if user_with_email and str(user_with_email.id) != str(user_id):
                raise DuplicateValueException(ERROR_MESSAGES.DUPLICATE_VALUE("Email"))

        if form_data.password is not None:
            form_data.password = get_password_hash(form_data.password)

        updated_user = await user_service.update_user_by_id(
            id=str(user_id), form_data=form_data
        )

        return ResponseModel(
            status_code=200,
            message=SUCCESS_MESSAGE.UPDATED,
            data=updated_user,
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/", response_model=ResponseModel)
async def get_all_file(
    current_user: Annotated[TokenData, Depends(get_not_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
    search: Optional[str] = Query(None),
):
    try:
        users = await user_service.get_users(skip=skip, limit=limit, search=search)

        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=users
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/me", response_model=ResponseModel)
async def get_user_login_data(
    current_user: Annotated[TokenData, Depends(get_verified_user)],
):
    try:
        del current_user.password
        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=current_user
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.delete("/{user_id}", response_model=ResponseModel)
async def delete_items(
    user_id: UUID,
    current_user: Annotated[TokenData, Depends(get_not_user)],
):
    try:
        is_user_exist = await user_service.get_user_by_id(str(user_id))

        if is_user_exist is None:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("Pengguna"))

        await user_service.delete_user_by_id(str(user_id))

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))
