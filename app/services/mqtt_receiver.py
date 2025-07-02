import paho.mqtt.client as mqtt
import json
import os

from app.env import (
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_RECV_TOPIC,
    MQTT_BROKER,
    MQTT_PORT,
)
from app.queue import add_to_queue, pop_next, load_queue
from app.services.conversation import conversation_service
from app.models.messages import FromMessage, MessageCreateModel
from app.services.message import message_service


# Saat berhasil terhubung ke broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Receiver connected to MQTT broker")
        client.subscribe(MQTT_RECV_TOPIC)
        print(f"📡 Subscribed to topic: {MQTT_RECV_TOPIC}")
    else:
        print("❌ Connection failed with code:", rc)


# Saat menerima pesan
async def on_message(client, userdata, msg):
    try:
        message = msg.payload.decode()
        print(f"\n📩 [INCOMING] {message}")

        if ">>" in message:
            raw_nomor, isi = message.split(">>")
            nomor = raw_nomor.strip().split()[0]
            isi = isi.strip()

            if not isi:
                print("⚠️ Pesan kosong, dilewati")
                return

            # Ganti awalan 62 dengan 0
            if nomor.startswith("62"):
                nomor = "0" + nomor[2:]

            conversation = await conversation_service.get_one_conversation_by_sender(
                nomor
            )

            if conversation is None:
                conversation = await conversation_service.create_new_conversation(
                    title="Chat from WhatsApp", sender=nomor
                )

            add_to_queue(
                {"nomor": nomor, "isi": isi, "conversation_id": conversation.id}
            )

            _new_chat_from_user = MessageCreateModel(
                **{
                    "message": message,
                    "conversation_id": str(conversation.id),
                    "from_message": FromMessage.USER,
                }
            )
            await message_service.create_new_message(_new_chat_from_user)

            print(f"📥 [QUEUED] {nomor} : {isi}")
        else:
            print("⚠️ Format pesan tidak sesuai")
    except Exception as e:
        print("❗ Error saat memproses pesan:", e)


# MQTT setup
client = mqtt.Client()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()
