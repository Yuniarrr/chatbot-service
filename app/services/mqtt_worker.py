import asyncio
import time
import hashlib
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import paho.mqtt.client as mqtt

from app.env import (
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_RECV_TOPIC,
    MQTT_SEND_TOPIC,
    MQTT_BROKER,
    MQTT_PORT,
)
from app.queue import add_to_queue, pop_next
from app.retrieval.chain import chain_service

client = mqtt.Client()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ MQTT connected")
        client.subscribe(MQTT_RECV_TOPIC)
        print(f"📡 Subscribed to: {MQTT_RECV_TOPIC}")
    else:
        print("❌ MQTT failed with code:", rc)


def on_message(client, userdata, msg):
    try:
        message = msg.payload.decode()
        print(f"\n📩 [INCOMING] {message}")

        if ">>" in message:
            raw_nomor, isi = message.split(">>")
            nomor = raw_nomor.strip().split()[0]
            isi = isi.strip()

            if nomor.startswith("62"):
                nomor = "0" + nomor[2:]

            add_to_queue({"nomor": nomor, "isi": isi})
            print(f"📥 [QUEUED] {nomor} : {isi}")
        else:
            print("⚠️ Format tidak sesuai")
    except Exception as e:
        print("❗ MQTT msg error:", e)


client.on_connect = on_connect
client.on_message = on_message


async def process_with_ai(message: str, sender: str):
    start_time = time.time()
    conversation_id = hashlib.md5(sender.encode()).hexdigest()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content_blocks = [{"type": "text", "text": message}]
    messages = [
        SystemMessage(content=f"Sender: {sender}. Timestamp: {now}."),
        HumanMessage(content=content_blocks),
    ]

    agent_executor = chain_service.create_agent("openai")
    response = await agent_executor.ainvoke(
        {"messages": messages},
        {"configurable": {"thread_id": conversation_id}},
    )

    ai_messages = [
        msg.content
        for msg in response["messages"]
        if isinstance(msg, AIMessage) and msg.content.strip()
    ]

    return ai_messages, time.time() - start_time


async def mqtt_loop():
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    while True:
        msg = pop_next()
        if msg:
            nomor = msg["nomor"].strip()
            isi = msg["isi"].strip()

            try:
                ai_contents, duration = await process_with_ai(isi, nomor)
                answer = f"{nomor} << {ai_contents[-1] if ai_contents else '(kosong)'}"

                print(f"🧪 Kirim ke {MQTT_SEND_TOPIC}")
                result = client.publish(MQTT_SEND_TOPIC, answer, qos=1)
                if result.rc == 0:
                    print(f"🔁 [REPLIED] {answer}")
                else:
                    print(f"❌ Gagal kirim (rc={result.rc})")
            except Exception as e:
                print("❗ Error:", e)
                fallback = f"{nomor} << Terjadi kesalahan sistem."
                client.publish(MQTT_SEND_TOPIC, fallback, qos=1)
        else:
            await asyncio.sleep(1)
