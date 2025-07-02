import asyncio
import time
import hashlib
import json
import sys
import os
from datetime import datetime
from typing import List
from aiomqtt import Client, MqttError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from app.models.messages import FromMessage, MessageCreateModel
from app.env import (
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_RECV_TOPIC,
    MQTT_SEND_TOPIC,
    MQTT_BROKER,
    MQTT_PORT,
)
from app.queue import add_to_queue, get_from_queue, pop_batch, pop_next, message_queue
from app.retrieval.chain import chain_service
from app.services.message import message_service


# Fix untuk event loop Windows
if sys.platform.lower() == "win32" or os.name.lower() == "nt":
    from asyncio import set_event_loop_policy, WindowsSelectorEventLoopPolicy

    set_event_loop_policy(WindowsSelectorEventLoopPolicy())


def limit_conversation_messages(
    messages: List[BaseMessage], max_count: int = 3
) -> List[BaseMessage]:
    if len(messages) <= max_count:
        return messages

    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    recent_messages = other_messages[-max_count:]

    return system_messages + recent_messages


async def process_with_ai(
    message: str, sender: str, conversation_id: str, max_recent_messages: int = 3
):
    start_time = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config = {"configurable": {"thread_id": conversation_id}}
    agent_executor = chain_service.create_agent("openai")

    try:
        current_state = await agent_executor.aget_state(config)
        existing_messages = current_state.values.get("messages", [])
        content_blocks = [{"type": "text", "text": message}]
        new_message = HumanMessage(content=content_blocks)

        all_messages = existing_messages + [new_message]
        limited_messages = limit_conversation_messages(
            all_messages, max_recent_messages
        )

        system_msg = SystemMessage(content=f"Sender: {sender}. Timestamp: {now}.")
        final_messages = [system_msg] + [
            msg for msg in limited_messages if not isinstance(msg, SystemMessage)
        ]

        response = await agent_executor.ainvoke({"messages": final_messages}, config)

    except Exception as e:
        print(f"Error getting conversation history: {e}")
        content_blocks = [{"type": "text", "text": message}]
        response = await agent_executor.ainvoke(
            {
                "messages": [
                    SystemMessage(content=f"Sender: {sender}. Timestamp: {now}."),
                    HumanMessage(content=content_blocks),
                ]
            },
            config,
        )

    ai_messages = [
        msg.content
        for msg in response["messages"]
        if isinstance(msg, AIMessage) and msg.content.strip()
    ]
    return ai_messages, time.time() - start_time


MAX_WORKERS = 5


async def process_message(client, msg):
    nomor = msg["nomor"].strip()
    conversation_id = msg["conversation_id"].strip()
    isi = msg["isi"].strip()

    if not isi:
        return

    try:
        await client.publish(
            MQTT_SEND_TOPIC, f"{nomor} << Pesan Anda sedang diproses...", qos=1
        )

        ai_contents, duration = await process_with_ai(isi, nomor, conversation_id)
        jawaban = ai_contents[-1] if ai_contents else "(kosong)"

        await client.publish(MQTT_SEND_TOPIC, f"{nomor} << {jawaban}", qos=1)
        await client.publish(
            MQTT_SEND_TOPIC,
            f"{nomor} << Total processing time: {duration:.2f} seconds",
            qos=1,
        )

        print(f"🔁 [REPLIED] {nomor} << {jawaban}")

        _new_chat_from_assistant = MessageCreateModel(
            message=jawaban,
            conversation_id=str(conversation_id),
            from_message=FromMessage.BOT,
        )
        await message_service.create_new_message(_new_chat_from_assistant)

    except Exception as e:
        print("❗ Error:", e)
        await client.publish(
            MQTT_SEND_TOPIC, f"{nomor} << Terjadi kesalahan sistem.", qos=1
        )


async def worker(client, id):
    while True:
        msg = await get_from_queue()
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ~ 👀 Checking queue entry: {msg}"
        )
        print(f"[Worker-{id}] 👷 Processing: {msg}")
        await process_message(client, msg)
        message_queue.task_done()


async def mqtt_responder_loop():
    try:
        async with Client(
            hostname=MQTT_BROKER,
            port=MQTT_PORT,
            username=MQTT_USERNAME,
            password=MQTT_PASSWORD,
        ) as client:
            print("✅ Connected to MQTT broker (responder)")
            print(f"📡 Ready to publish to: {MQTT_SEND_TOPIC}")

            # Start worker pool
            workers = [
                asyncio.create_task(worker(client, i)) for i in range(MAX_WORKERS)
            ]

            # Worker pool jalan terus...
            await message_queue.join()  # tunggu antrian kosong

            # (opsional) cancel semua worker jika sudah selesai
            for w in workers:
                w.cancel()

    except MqttError as error:
        print(f"❌ MQTT error: {error}")


if __name__ == "__main__":
    asyncio.run(mqtt_responder_loop())
