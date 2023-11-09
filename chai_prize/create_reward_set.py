import os
import json
import random
from pathlib import Path
from collections import defaultdict

import fire
from datasets import load_dataset
from tqdm import tqdm

from chai_prize.util.data import (
    has_bot_message,
    calc_max_length,
    shrink,
    has_correct_roles,
    is_bad_chat
)
from chai_prize.datasets.chai import (
    parse_chai_conversation,
    is_whitelisted_model,
    is_good_feedback
)


def process_chai(
    sample_rate: float = 1.0,
    dataset_name: str = "ChaiML/20231007_chai_prize_model_feedback_all",
    character_dataset_name: str = "ChaiML/seasonIII_chatAI_configurations",
    max_length: int = 20000,
    min_messages: int = 6,
    only_whitelist: bool = False,
    only_thumbs_up: bool = False,
    only_good_feedback: bool = False
):
    records = []
    characters = {row["bot_id"]: row for row in load_dataset(character_dataset_name, split="train")}
    for row in tqdm(load_dataset(dataset_name, split="train")):
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

        if only_thumbs_up and not row["thumbs_up"]:
            continue

        if only_good_feedback and not is_good_feedback(row["feedback"]):
            continue

        if only_whitelist and not is_whitelisted_model(row["model_name"]):
            continue

        if not has_bot_message(chat):
            continue

        if len(chat) < min_messages:
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
        shrinked_chat = shrink(chat, max_length)
        is_shrinked = len(shrinked_chat) < len(chat)
        chat = shrinked_chat

        records.append({
            "messages": chat,
            "char_name": char_name,
            "original_fields": row,
            "source": "chai",
            "is_shrinked": is_shrinked,
            "bot_id": bot_id
        })

    print("From Chai count:", len(records))
    if records:
        print("Chai max length:", calc_max_length(records))
    return records


def is_close(message1, message2):
    tokens1 = set(message1.split())
    tokens2 = set(message2.split())
    min_count = min(len(tokens1), len(tokens2))
    intersection = tokens1.intersection(tokens2)
    return len(intersection) / min_count > 0.7


def main(config_path, output_dir):
    random.seed(42)
    with open(config_path) as r:
        config = json.load(r)
    chai_records = process_chai(**config.get("chai"))

    final_records = []
    if config["mode"] == "deleted":
        for record in chai_records:
            chat = record["messages"]
            if sum([m.get("is_deleted", False) for m in chat]) == 0:
                continue

            deleted_messages = []
            prev_messages = []
            for message in chat:
                if message.get("is_deleted", False):
                    deleted_messages.append(message)
                    continue

                if len(deleted_messages) >= 1 and len(deleted_messages) <= 2:
                    for rejected_message in deleted_messages:
                        assert prev_messages[-1]["role"] == "user"
                        if rejected_message == message["content"]:
                            continue
                        if not message["content"].strip():
                            continue
                        if not rejected_message["content"].strip():
                            continue
                        if is_close(message["content"], rejected_message["content"]):
                            continue
                        chosen_messages = prev_messages + [message]
                        rejected_messages = prev_messages + [rejected_message]
                        final_records.append({
                            "rejected_messages": rejected_messages,
                            "chosen_messages": chosen_messages,
                            "char_name": record["char_name"]
                        })

                deleted_messages = []
                prev_messages.append(message)
    elif config["mode"] == "thumbs_up":
        char_records = defaultdict(list)
        for record in chai_records:
            record["messages"] = [m for m in record["messages"] if not m.get("is_deleted", False)]
            char_records[record["bot_id"]].append(record)

        for _, records in char_records.items():
            positive_records = [r for r in records if r["original_fields"]["thumbs_up"]]
            negative_records = [r for r in records if not r["original_fields"]["thumbs_up"]]
            if not positive_records or not negative_records:
                continue
            for pos_record, neg_record in zip(positive_records, negative_records):
                final_records.append({
                    "chosen_messages": pos_record["messages"],
                    "rejected_messages": neg_record["messages"],
                    "char_name": pos_record["char_name"]
                })
    elif config["mode"] == "thumbs_up_pointwise":
        for conv_id, record in enumerate(chai_records):
            messages = [m for m in record["messages"] if not m.get("is_deleted", False)]
            if not messages:
                continue
            final_records.append({
                "messages": messages,
                "target": int(record["original_fields"]["thumbs_up"]),
                "char_name": record["char_name"],
                "conv_id": conv_id
            })
    else:
        char_records = defaultdict(list)
        for conv_id, record in enumerate(chai_records):
            messages = record["messages"][:]
            if not messages:
                continue
            while messages[-1]["role"] != "bot" and not messages[-1].get("is_deleted", False):
                messages.pop()
            examples = []
            current_context = []
            for i, message in enumerate(messages):
                if message.get("is_deleted", False):
                    examples.append({
                        "messages": current_context + [message],
                        "char_name": record["char_name"],
                        "conv_id": conv_id,
                        "target": 0
                    })
                    continue
                current_context.append(message)
                if message["role"] == "bot" and i >= 6:
                    examples.append({
                        "messages": current_context[:],
                        "conv_id": conv_id,
                        "char_name": record["char_name"],
                        "target": 1
                    })
            if not record["is_shrinked"] and len(examples) > 1:
                examples[-1]["target"] = 0
            final_records.extend(examples)

    print("All count after cleaning:", len(final_records))

    char_names = list({r["char_name"] for r in final_records})
    random.shuffle(char_names)
    border = int(0.95 * len(char_names))
    train_chars = set(char_names[:border])
    val_chars = set(char_names[border:])
    train_records = [r for r in final_records if r["char_name"] in train_chars]
    val_records = [r for r in final_records if r["char_name"] in val_chars]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, "train.jsonl"), "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(os.path.join(output_dir, "val.jsonl"), "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
