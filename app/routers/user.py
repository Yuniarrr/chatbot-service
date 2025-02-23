import logging

from typing import Annotated
from fastapi import APIRouter, Depends, Query

from app.core.response import ResponseModel
from app.models.users import AddUserForm, Role, LoginForm
from app.services.user import user_service
from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.utils.auth import get_password_hash, TokenData, get_admin_user, get_not_user
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
        )

        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.CREATED, data=new_user
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.delete("/", response_model=ResponseModel)
async def delete_items(
    current_user: Annotated[TokenData, Depends(get_not_user)],
    ids: list[str] = Query(description="List of user IDs to delete"),
):
    try:
        for user_id in ids:
            is_user_exist = await user_service.get_user_by_id(user_id)

            if is_user_exist is None:
                raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("User"))

            await user_service.delete_user_by_id(user_id)

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))
