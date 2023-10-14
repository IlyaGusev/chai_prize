import os
import json
import copy
from collections import Counter
import random
from pathlib import Path
from statistics import median

import fire
from datasets import load_dataset
from tqdm import tqdm

from chai_prize.util.data import (
    revert_flattening,
    has_bot_message,
    clean_bot_message,
    calc_max_length,
    shrink,
    has_repetition,
    has_correct_roles
)
from chai_prize.datasets.chai import parse_chai_conversation


def is_whitelisted_model(model_name):
    models = (
        "ilyagusev",
        "khoantap",
        "anhnv125",
        "tokenbender",
        "tehvenom",
        "khanhnto",
        "liquac09",
        "monkeydddd",
        "alkahestry",
        "the-face-of-goonery",
        "jnewber",
        "chargoddard",
        "ansoi"
    )
    for model in models:
        if model in model_name:
            return True
    return False


def process_chai(
    sample_rate: float = 1.0,
    dataset_name: str = "ChaiML/20231007_chai_prize_model_feedback_all",
    character_dataset_name: str = "ChaiML/seasonIII_chatAI_configurations",
    max_length: int = 20000,
    min_messages: int = 4,
    only_whitelist: bool = False
):
    records = []
    ctrl_counts = Counter()
    characters = {row["bot_id"]: row for row in load_dataset(character_dataset_name, split="train")}
    for row in tqdm(load_dataset(dataset_name, split="train")):
        conversation_id = row["conversation_id"]
        bot_id = row["bot_id"]

        if bot_id not in characters:
            continue

        text = row["conversation"]
        if "INST" in text or "START" in text:
            continue

        char_name = text.split(":")[0].strip()
        chat = list(parse_chai_conversation(text))
        if not chat:
            continue

        if not has_bot_message(chat):
            continue

        if len(chat) < min_messages:
            continue

        if only_whitelist and not is_whitelisted_model(row["model_name"]):
            continue

        if not has_correct_roles(chat):
            continue

        if random.random() > sample_rate:
            continue

        character = characters[bot_id]

        memory = character["memory"]
        memory = memory if memory else ""
        memory = memory.strip()

        prompt = character["prompt"]
        prompt = prompt if prompt else ""
        prompt = prompt.strip()

        system_chat = [{"role": "system", "content": memory}, {"role": "prompt", "content": prompt}]
        chat = system_chat + chat
        chat = shrink(chat, max_length)

        records.append({
            "messages": chat,
            "char_name": char_name,
            "original_fields": row,
            "source": "chai"
        })

    print("From Chai count:", len(records))
    if records:
        print("Chai max length:", calc_max_length(records))
    return records


def main(config_path, output_dir):
    random.seed(42)
    with open(config_path) as r:
        config = json.load(r)
    chai_records = process_chai(**config.get("chai"))

    records = []
    for record in chai_records:
        chat = record["messages"]
        if sum([m.get("is_deleted", False) for m in chat]) == 0:
            continue

        deleted_messages = []
        prev_messages = []
        for message in chat:
            if message.get("is_deleted", False):
                deleted_messages.append(message["content"])
                continue

            if len(deleted_messages) >= 1 and len(deleted_messages) <= 2:
                for rejected_message in deleted_messages:
                    assert prev_messages[-1]["role"] == "user"
                    if rejected_message == message["content"]:
                        continue
                    records.append({
                        "context": prev_messages[:],
                        "rejected_message": rejected_message,
                        "chosen_message": message["content"],
                        "char_name": record["char_name"]
                    })

            deleted_messages = []
            prev_messages.append(message)
    print("All count after cleaning:", len(records))

    random.shuffle(records)
    border = int(0.95 * len(records))
    train_records = records[:border]
    val_records = records[border:]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, "train.jsonl"), "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(os.path.join(output_dir, "val.jsonl"), "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
