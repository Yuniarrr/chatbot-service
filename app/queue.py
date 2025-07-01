import os
import json

from app.env import QUEUE_FILE


def load_queue():
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            return json.load(f)
    return []


def save_queue(queue):
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def add_to_queue(entry):
    queue = load_queue()
    queue.append(entry)
    save_queue(queue)


def pop_next():
    queue = load_queue()
    if queue:
        next_msg = queue.pop(0)
        save_queue(queue)
        return next_msg
    return None
