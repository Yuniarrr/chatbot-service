import logging

from fastapi import APIRouter
from fastcrud.exceptions.http_exceptions import (
    CustomException,
    NotFoundException,
    UnauthorizedException,
    DuplicateValueException,
)


from app.core.response import ResponseModel
from app.models.users import RegisterForm, Role, LoginForm
from app.services.user import user_service
from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.utils.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
)
from app.core.logger import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/register", response_model=ResponseModel)
async def register(form_data: RegisterForm):
    try:
        is_email_exist = await user_service.get_user_by_email(form_data.email)

        if is_email_exist is None:
            raise DuplicateValueException(ERROR_MESSAGES.DUPLICATE_VALUE("Email"))

        hash_password = get_password_hash(form_data.password)

        new_user = await user_service.create_new_user(
            full_name=form_data.full_name,
            email=form_data.email,
            password=hash_password,
            role=Role.USER,
        )

        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.CREATED, data=new_user
        )
    except Exception as e:
        raise CustomException(500, str(e))


@router.post("/login", response_model=ResponseModel)
async def login(form_data: LoginForm):
    try:
        is_email_exist = await user_service.get_user_by_email(form_data.email)

        if is_email_exist is None:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("Email"))

        is_verify = verify_password(form_data.password, is_email_exist.password)

        if is_verify is False:
            raise UnauthorizedException(ERROR_MESSAGES.UNAUTHORIZED)

        access_token = create_access_token(
            data={
                "email": is_email_exist.email,
                "id": is_email_exist.id,
                "role": is_email_exist.role,
            }
        )

        refresh_token = create_refresh_token(
            data={
                "email": is_email_exist.email,
                "id": is_email_exist.id,
                "role": is_email_exist.role,
            }
        )

        del is_email_exist.password

        return ResponseModel(
            status_code=201,
            message=SUCCESS_MESSAGE.CREATED,
            data={
                "access_token": access_token,
                "refresh_token": refresh_token,
                **is_email_exist.model_dump(),
            },
        )
    except Exception as e:
        raise CustomException(500, str(e))
