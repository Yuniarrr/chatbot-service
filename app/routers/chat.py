import json
import base64
import logging
import uuid
import os

from typing import Optional
from fastapi import APIRouter, Form, UploadFile, File, Depends, Request
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
from app.services.uploader import uploader_service
from app.env import ASSET_URL
from app.utils.twillio_util import download_twilio_media

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/", response_model=ResponseModel)
async def chat_to_assistant(
    # form_data: MessageForm,
    message: str = Form(...),
    conversation_id: uuid.UUID = Form(...),
    collection_name: Optional[str] = Form("administration"),
    model: Optional[str] = Form("ollama"),
    file: Optional[UploadFile] = File(None),
    current_user=Depends(get_verified_user),
):
    try:
        file_data = ""
        filename = ""
        if file:
            unsanitized_filename = file.filename
            filename = os.path.basename(unsanitized_filename)
            id = str(uuid.uuid4())
            filename = f"{id}_{filename}"
            contents = await file.read()
            _, file_path = uploader_service.upload_to_local(contents, filename)
            with open(file_path, "rb") as file_content:
                file_data = base64.b64encode(file_content.read()).decode("utf-8")

        async def chain_streamer(
            message: str,
            conversation_id: uuid.UUID,
            collection_name: str,
            model: str,
            current_user: UserReadWithPasswordModel,
            file: Optional[UploadFile] = None,
        ):
            _new_chat_from_user = MessageCreateModel(
                **{
                    "message": message,
                    "conversation_id": conversation_id,
                    "from_message": FromMessage.USER,
                    **(
                        {
                            "file_url": f"{ASSET_URL}/{filename}",
                            "file_size": file.size,
                            "file_name": filename,
                        }
                        if file
                        else {}
                    ),
                }
            )

            await message_service.create_new_message(_new_chat_from_user)

            accumulated_text = ""

            agent_executor = chain_service.create_agent(model)

            content_blocks = [{"type": "text", "text": message}]
            if file and file.content_type.startswith("image/"):
                content_blocks.append(
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": file_data,
                        "mime_type": file.content_type,
                    }
                )
            elif file and file.content_type.startswith("application/"):
                content_blocks.append(
                    {
                        "type": "file",
                        "source_type": "base64",
                        "data": file_data,
                        "mime_type": file.content_type,
                        "filename": filename,
                    }
                )

            messages = [
                SystemMessage(
                    content=f"User ID atau sender pesan adalah: {current_user.id}"
                ),
                HumanMessage(content=content_blocks),
            ]

            async for step in agent_executor.astream(
                {
                    "messages": messages,
                    "collection_name": collection_name,
                    "user_id": current_user.id,
                },
                {"configurable": {"thread_id": conversation_id}},
                stream_mode="values",
            ):
                print("step")
                # print(step)
                accumulated_text = step["messages"][-1].content
                yield step["messages"][-1].content

            _new_chat_from_assistant = MessageCreateModel(
                **{
                    "message": accumulated_text,
                    "conversation_id": conversation_id,
                    "from_message": FromMessage.BOT,
                }
            )

            await message_service.create_new_message(_new_chat_from_assistant)

        return EventSourceResponse(
            chain_streamer(
                message, conversation_id, collection_name, model, current_user, file
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.post("/message")
async def reply(request: Request):
    try:
        form_data = await request.form()
        print(form_data)
        message = form_data.get("Body")
        sender = form_data.get("From")
        # receiver = form_data.get("To")

        print("message")
        print(message)
        print(form_data.get("MediaUrl0"))

        media = None
        if form_data.get("MediaUrl0") is not None:
            media = await download_twilio_media(form_data.get("MediaUrl0"))

        conversation = await conversation_service.get_conversation_by_sender(sender)

        if conversation == None:
            conversation = await conversation_service.create_new_conversation(
                title="Chat from WhatsApp", sender=sender
            )

        agent_executor = chain_service.create_agent("openai")

        content_blocks = [{"type": "text", "text": message}]
        if media and media["content_type"].startswith("image/"):
            content_blocks.append(
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": media["file_data"],
                    "mime_type": media["content_type"],
                    "image_url": {"url": media["final_url"]},
                }
            )
        elif media and media["content_type"].startswith("application/"):
            content_blocks.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": media["file_data"],
                    "mime_type": media["content_type"],
                    "filename": media["filename"],
                }
            )

        _new_chat_from_user = MessageCreateModel(
            **{
                "message": message,
                "conversation_id": str(conversation.id),
                "from_message": FromMessage.USER,
                **(
                    {
                        "file_url": f"{ASSET_URL}/{media['filename']}",
                        "file_size": media["file_size"],
                        "file_name": media["filename"],
                    }
                    if media
                    else {}
                ),
            }
        )

        await message_service.create_new_message(_new_chat_from_user)

        messages = [
            SystemMessage(
                content=f"User ID atau sender pesan adalah: {sender}."
                + (
                    f" Gunakan url berikut sebagai image_url, jika akan menambahkan atau menyimpan data ke opportunity {ASSET_URL + '/' + media['filename']}"
                    if media
                    else ""
                )
            ),
            HumanMessage(content=content_blocks),
        ]

        response = await agent_executor.ainvoke(
            {
                "messages": messages,
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

        _new_chat_from_assistant = MessageCreateModel(
            **{
                "message": ai_content,
                "conversation_id": str(conversation.id),
                "from_message": FromMessage.BOT,
            }
        )

        await message_service.create_new_message(_new_chat_from_assistant)

        resp = MessagingResponse()
        resp.message(ai_content)
        return Response(content=str(resp), media_type="application/xml")
    except Exception as e:
        log.exception("Error processing message")
        print(str(e))
        error_resp = MessagingResponse()
        error_resp.message("Terjadi kesalahan dalam sistem. Tolong hubungi developer.")
        return Response(content=str(error_resp), media_type="application/xml")
