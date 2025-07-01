import asyncio
import time
import hashlib
import json
from datetime import datetime
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
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
client.loop_start()
# client.on_message = on_message

def limit_conversation_messages(messages: List[BaseMessage], max_count: int = 3) -> List[BaseMessage]:
    """
    Fungsi helper untuk membatasi pesan percakapan
    """
    if len(messages) <= max_count:
        return messages
    
    # Pisahkan system messages dan messages lainnya
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    
    # Ambil pesan terakhir sesuai limit (tidak termasuk system messages)
    recent_messages = other_messages[-max_count:]
    
    return system_messages + recent_messages

async def process_with_ai(message: str, sender: str, max_recent_messages: int = 3):
    start_time = time.time()
    conversation_id = hashlib.md5(sender.encode()).hexdigest()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    config = {"configurable": {"thread_id": conversation_id}}
    agent_executor = chain_service.create_agent("openai")

    try:
        # Ambil state saat ini untuk mendapatkan riwayat pesan
        current_state = await agent_executor.aget_state(config)
        existing_messages = current_state.values.get("messages", [])
        
        # Buat pesan baru
        content_blocks = [{"type": "text", "text": message}]
        new_message = HumanMessage(content=content_blocks)
        
        # Gabungkan dengan pesan yang ada
        all_messages = existing_messages + [new_message]
        
        # LIMIT PESAN: Hanya ambil pesan terakhir
        limited_messages = limit_conversation_messages(all_messages, max_recent_messages)
        
        # Tambahkan system message dengan info terbaru
        system_msg = SystemMessage(content=f"Sender: {sender}. Timestamp: {now}.")
        final_messages = [system_msg] + [msg for msg in limited_messages if not isinstance(msg, SystemMessage)]
        
        print(f"Sending {len(final_messages)} messages to agent (limited from {len(all_messages)})")
        
        # Invoke dengan pesan yang sudah dibatasi
        response = await agent_executor.ainvoke(
            {"messages": final_messages},
            config,
        )
        
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        # Fallback: kirim hanya pesan baru
        content_blocks = [{"type": "text", "text": message}]
        messages = [
            SystemMessage(content=f"Sender: {sender}. Timestamp: {now}."),
            HumanMessage(content=content_blocks),
        ]
        
        response = await agent_executor.ainvoke(
            {"messages": messages},
            config,
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

            if not isi:
                continue

            try:
                ai_contents, duration = await process_with_ai(isi, nomor)
                answer = f"{nomor} << {ai_contents[-1] if ai_contents else '(kosong)'}"
                answer_duration = (
                    f"{nomor} << Total processing time: {duration:.2f} seconds"
                )

                print(f"🧪 Kirim ke {MQTT_SEND_TOPIC}")
                result = client.publish(MQTT_SEND_TOPIC, answer, qos=1)
                result = client.publish(MQTT_SEND_TOPIC, answer_duration, qos=1)
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
