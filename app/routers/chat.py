import json
import logging
import uuid

from fastapi import APIRouter, UploadFile, File, Depends, Request
from sse_starlette import EventSourceResponse
from langchain_core.messages import HumanMessage, SystemMessage
from twilio.twiml.messaging_response import MessagingResponse, Message

from app.core.constants import SUCCESS_MESSAGE
from app.core.exceptions import InternalServerException
from app.core.response import ResponseModel
from app.core.logger import SRC_LOG_LEVELS
from app.models.messages import FromMessage, MessageCreateModel, MessageForm
from app.utils.auth import (
    TokenData,
    get_verified_user,
)
from app.services.message import message_service
from app.retrieval.chain import AgentState, chain_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/", response_model=ResponseModel)
async def chat_to_assistant(
    form_data: MessageForm,
    # file: Optional[UploadFile] = File(None),
    user=Depends(get_verified_user),
):
    try:

        async def chain_streamer(data: MessageForm):
            _new_chat_from_user = MessageCreateModel(
                **{
                    "message": data.message,
                    "conversation_id": data.conversation_id,
                    "from_message": FromMessage.USER,
                }
            )

            await message_service.create_new_message(_new_chat_from_user)

            accumulated_text = ""

            agent_executor = chain_service.create_agent(data.model)
            system_prompt = chain_service.agent_system_prompt()

            for step in agent_executor.stream(
                {
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=data.message),
                    ],
                    "collection_name": data.collection_name,
                },
                stream_mode="values",
            ):
                print("step")
                print(step)
                yield step["messages"][-1].content

            _new_chat_from_assistant = MessageCreateModel(
                **{
                    "message": data.message,
                    "conversation_id": data.conversation_id,
                    "from_message": FromMessage.USER,
                }
            )

            await message_service.create_new_message(_new_chat_from_assistant)

        return EventSourceResponse(
            chain_streamer(form_data),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.post("/message")
async def reply(request: Request):
    try:
        form_data = await request.form()
        msg = form_data.get("Body")
        print(msg)

        resp = MessagingResponse()
        resp.message("Bismillah")
        print("response")
        print(resp)
        return str(resp)
    except Exception as e:
        raise InternalServerException(str(e))
