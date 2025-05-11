import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Form, Query, Request, UploadFile, File, Depends

from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.core.exceptions import InternalServerException, NotFoundException
from app.core.logger import SRC_LOG_LEVELS
from app.core.response import ResponseModel
from app.models.opportunities import OpportunitiesForm, OpportunityType
from app.utils.auth import TokenData, get_verified_user, get_not_user
from app.services.opportunity import opportunity_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/", response_model=ResponseModel)
async def add_new_opportunity(
    current_user: Annotated[TokenData, Depends(get_verified_user)],
    form_data: OpportunitiesForm,
):
    try:
        new_opportunity = await opportunity_service.insert_new_opportunity(
            form_data=form_data, uploader=current_user.full_name
        )

        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.CREATED, data=new_opportunity
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/", response_model=ResponseModel)
async def get_all_opportunity(
    current_user: Annotated[TokenData, Depends(get_verified_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
    type: Optional[OpportunityType] = None,
):
    try:
        opportunities = await opportunity_service.get_opportunities(
            skip=skip, limit=limit, type=type
        )

        return ResponseModel(
            status_code=200,
            message=SUCCESS_MESSAGE.RETRIEVED,
            data=opportunities["data"],
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/{opportunity_id}", response_model=ResponseModel)
async def get_opportunity_by_id(
    opportunity_id: str,
    current_user: Annotated[TokenData, Depends(get_verified_user)],
):
    try:
        opportunity = await opportunity_service.get_opportunity_by_id(opportunity_id)
        if not opportunity:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("opportunity"))

        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=opportunity
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.patch("/{opportunity_id}", response_model=ResponseModel)
async def update_opportunity_by_id(
    opportunity_id: str,
    current_user: Annotated[TokenData, Depends(get_verified_user)],
    form_data: OpportunitiesForm,
):
    try:
        opportunity = await opportunity_service.get_opportunity_by_id(opportunity_id)
        if not opportunity:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("opportunity"))

        updated_opportunity = await opportunity_service.update_opportunity_by_id(
            opportunity_id=opportunity_id, form_data=form_data
        )

        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.CREATED, data=updated_opportunity
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.delete("/{opportunity_id}", response_model=ResponseModel)
async def delete_opportunity_by_id(
    opportunity_id: str,
    current_user: Annotated[TokenData, Depends(get_not_user)],
):
    try:
        opportunity = await opportunity_service.get_opportunity_by_id(opportunity_id)
        if not opportunity:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("opportunity"))

        await opportunity_service.delete_opportunity_by_id(opportunity_service)

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))
