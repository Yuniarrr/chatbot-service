import sys
import os
import asyncio
from aiomqtt import Client, MqttError

from app.env import (
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_RECV_TOPIC,
    MQTT_BROKER,
    MQTT_PORT,
)
from app.queue import add_to_queue
from app.services.conversation import conversation_service
from app.models.messages import FromMessage, MessageCreateModel
from app.services.message import message_service


async def handle_message(message):
    try:
        payload = message.payload.decode()
        print(f"\n📩 [INCOMING] {payload}")

        if ">>" in payload:
            raw_nomor, isi = payload.split(">>")
            nomor = raw_nomor.strip().split()[0]
            isi = isi.strip()

            if not isi:
                print("⚠️ Pesan kosong, dilewati")
                return

            if nomor.startswith("62"):
                nomor = "0" + nomor[2:]

            conversation = await conversation_service.get_today_conversation_by_sender(
                nomor
            )

            if conversation is None:
                conversation = await conversation_service.create_new_conversation(
                    title="Chat from WhatsApp", sender=nomor
                )

            add_to_queue(
                {"nomor": nomor, "isi": isi, "conversation_id": str(conversation.id)}
            )

            _new_chat_from_user = MessageCreateModel(
                message=isi,
                conversation_id=str(conversation.id),
                from_message=FromMessage.USER,
            )
            await message_service.create_new_message(_new_chat_from_user)

            print(f"📥 [QUEUED] {nomor} : {isi}")
        else:
            print("⚠️ Format pesan tidak sesuai")
    except Exception as e:
        print("❗ Error saat memproses pesan:", e)


async def mqtt_receiver_loop():
    async with Client(
        hostname=MQTT_BROKER,
        port=MQTT_PORT,
        username=MQTT_USERNAME,
        password=MQTT_PASSWORD,
    ) as client:
        print("✅ Connected to MQTT broker (aiomqtt)")
        await client.subscribe(MQTT_RECV_TOPIC)
        print(f"📡 Subscribed to topic: {MQTT_RECV_TOPIC}")

        async for message in client.messages:
            asyncio.create_task(handle_message(message))
