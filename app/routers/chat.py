import json
import base64
import logging
import uuid
import os
import asyncio
import time

from twilio.rest import Client
from typing import Optional
from fastapi import APIRouter, Form, UploadFile, File, Depends, Request, BackgroundTasks
from sse_starlette import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from twilio.twiml.messaging_response import MessagingResponse, Message
from fastapi.responses import Response

from app.core.constants import ERROR_MESSAGES
from app.core.exceptions import InternalServerException, NotFoundException
from app.core.response import ResponseModel
from app.core.logger import SRC_LOG_LEVELS
from app.models.messages import FromMessage, MessageCreateModel
from app.models.users import UserReadWithPasswordModel
from app.utils.auth import get_verified_user
from app.services.message import message_service
from app.retrieval.chain import chain_service
from app.services.conversation import conversation_service
from app.services.uploader import uploader_service
from app.env import ASSET_URL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
from app.utils.split import split_by_newline_before_limit
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
        conversation = await conversation_service.get_conversation_by_id(
            str(conversation_id)
        )
        if not conversation:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("conversation"))

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
                # {"configurable": {"thread_id": conversation_id}},
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
async def reply(request: Request, background_tasks: BackgroundTasks):
    form_data = await request.form()
    print(form_data)

    # Quick ACK for Twilio webhook
    fast_resp = MessagingResponse()
    fast_resp.message("Pesan Anda sedang diproses...")
    response = Response(content=str(fast_resp), media_type="application/xml")

    # Trigger background processing
    background_tasks.add_task(process_in_background, form_data)

    return response


async def process_in_background(form_data):
    message = form_data.get("Body")
    sender = form_data.get("From")
    to = form_data.get("To")
    media_url = form_data.get("MediaUrl0")
    print("form_data")
    print(form_data)
    try:
        start_time = time.time()

        # Parallel media download and conversation fetch
        media_task = download_twilio_media(media_url) if media_url else asyncio.sleep(0)
        conversation_task = conversation_service.get_one_conversation_by_sender(sender)
        media, conversation = await asyncio.gather(media_task, conversation_task)

        if conversation is None:
            conversation = await conversation_service.create_new_conversation(
                title="Chat from WhatsApp", sender=sender
            )

        agent_executor = chain_service.create_agent("openai")

        content_blocks = [{"type": "text", "text": message}]
        if isinstance(media, dict) and media.get("content_type", "").startswith(
            "image/"
        ):
            content_blocks.append(
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": media["file_data"],
                    "mime_type": media["content_type"],
                    "image_url": {"url": media["final_url"]},
                }
            )
        elif isinstance(media, dict) and media.get("content_type", "").startswith(
            "application/"
        ):
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
                    if media and isinstance(media, dict)
                    else {}
                ),
            }
        )
        await message_service.create_new_message(_new_chat_from_user)

        messages = [
            SystemMessage(
                content=f"User ID atau sender pesan adalah: {sender}."
                + (
                    f"User telah mengunggah gambar yang mungkin berisi informasi untuk disimpan. "
                    f"Gunakan URL ini sebagai image_url jika perlu menyimpan data: {ASSET_URL}/{media['filename']}."
                    if media and isinstance(media, dict)
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

        ai_messages = [
            msg.content
            for msg in response["messages"]
            if isinstance(msg, AIMessage) and msg.content.strip()
        ]

        ai_content = (
            ai_messages[-1]
            if ai_messages
            else "Terjadi kesalahan, tidak ada respon dari AI. Tolong hubungi developer."
        )

        _new_chat_from_assistant = MessageCreateModel(
            **{
                "message": ai_content,
                "conversation_id": str(conversation.id),
                "from_message": FromMessage.BOT,
            }
        )
        await message_service.create_new_message(_new_chat_from_assistant)

        # Send AI response via Twilio
        # MAX_LENGTH = 1599
        # truncated_content = ai_content[:MAX_LENGTH]
        # await asyncio.to_thread(send_whatsapp_message, sender, truncated_content, to)

        contents = split_by_newline_before_limit(ai_content)

        for part in contents:
            print("part")
            print(part)
            await asyncio.to_thread(send_whatsapp_message, sender, part, to)

        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        log.exception("Error processing message")
        print("Exception:", str(e))
        await asyncio.to_thread(
            send_whatsapp_message,
            sender,
            "Terjadi kesalahan dalam sistem. Tolong hubungi developer.",
            to,
        )


# Twilio messaging function
def send_whatsapp_message(to: str, body: str, from_: str):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=body,
        from_=from_,
        to=to,
    )
