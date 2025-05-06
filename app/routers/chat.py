import json
import logging
import uuid

from fastapi import APIRouter, UploadFile, File, Depends, Request
from sse_starlette import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from twilio.twiml.messaging_response import MessagingResponse, Message
from fastapi.responses import Response

from app.core.exceptions import InternalServerException
from app.core.response import ResponseModel
from app.core.logger import SRC_LOG_LEVELS
from app.models.messages import FromMessage, MessageCreateModel, MessageForm
from app.models.users import UserReadWithPasswordModel
from app.utils.auth import (
    TokenData,
    get_verified_user,
)
from app.services.message import message_service
from app.retrieval.chain import chain_service
from app.services.conversation import conversation_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/", response_model=ResponseModel)
async def chat_to_assistant(
    form_data: MessageForm,
    # file: Optional[UploadFile] = File(None),
    current_user=Depends(get_verified_user),
):
    try:

        async def chain_streamer(
            data: MessageForm, current_user: UserReadWithPasswordModel
        ):
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

            async for step in agent_executor.astream(
                {
                    "messages": [
                        HumanMessage(content=data.message),
                    ],
                    "collection_name": data.collection_name,
                    "user_id": current_user.id,
                },
                {"configurable": {"thread_id": data.conversation_id}},
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
            chain_streamer(form_data, current_user),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.post("/message")
async def reply(request: Request):
    try:
        form_data = await request.form()
        message = form_data.get("Body")
        sender = form_data.get("From")
        # receiver = form_data.get("To")

        print("message")
        print(message)

        conversation = await conversation_service.get_conversation_by_sender(sender)

        if conversation == None:
            conversation = await conversation_service.create_new_conversation(
                title="Chat from WhatsApp", sender=sender
            )

        agent_executor = chain_service.create_agent("gemini")

        response = await agent_executor.ainvoke(
            {
                "messages": [
                    SystemMessage(
                        content=f"User ID atau sender pesan adalah: {sender}"
                    ),
                    HumanMessage(content=message),
                ],
            },
            {"configurable": {"thread_id": str(conversation.id)}},
        )

        print("response")
        print(response)

        messages = response["messages"]

        ai_messages = [
            message.content
            for message in messages
            if isinstance(message, AIMessage) and message.content.strip() != ""
        ]

        ai_content = (
            ai_messages[-1]
            if ai_messages
            else "Terjadi kesalahan, tidak ada respon dari AI. Tolong hubungi developer."
        )

        print("ai_content")
        print(ai_content)

        resp = MessagingResponse()
        resp.message(ai_content)
        return Response(content=str(resp), media_type="application/xml")
    except Exception as e:
        log.exception("Error processing message")
        print(str(e))
        error_resp = MessagingResponse()
        error_resp.message("Terjadi kesalahan dalam sistem. Tolong hubungi developer.")
        return Response(content=str(error_resp), media_type="application/xml")
