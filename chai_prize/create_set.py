import os
import json
import sys
import random
import fire
from pathlib import Path
from copy import deepcopy
from datasets import load_dataset
from tqdm import tqdm
from itertools import chain

def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])


def build_rpr_char_system_messages(char):
    name = char["name"]
    greeting = char["greeting"]
    context = char["context"]
    example_dialogue = char["example_dialogue"]

    full_context = ""
    if random.random() < 0.5:
        full_context += f"You are {name}. "
    full_context += f"{context}"

    chat = []
    if random.random() < 0.2:
        full_context += f"\nYour greeting: {greeting}"
        chat.append({
            "role": "bot",
            "content": greeting
        })
    if random.random() < 0.2:
        mapping = {
            "user": "user",
            "char": "bot"
        }
        example_messages = [f'{mapping[m["role"]]}: {m["content"]}' for m in example_dialogue]
        full_context += "\nDialogue example:\n" + "\n".join(example_messages)

    role = "system"
    if random.random() < 0.2:
        role = "prompt"
    chat.insert(0, {
        "role": role,
        "content": full_context
    })
    return chat


def process_rpr_row(row):
    for dialogue in row["dialogues"]:
        chat = dialogue["chat"]
        for message in chat:
            if message["role"] == "char":
                 message["role"] = "bot"
            if message["role"] == "operator":
                message["role"] = "user"

        system_messages = build_rpr_char_system_messages(row)
        chat = system_messages + chat
        yield {
            "messages": chat,
            "creator": dialogue["model_name"],
            "char_name": row["name"],
            "source": "rpr"
        }


def process_rpr(
    split: str = "en",
    sample_rate: float = 1.0,
    force_gpt4: bool = True
):
    records = []
    for row in tqdm(load_dataset("IlyaGusev/gpt_roleplay_realm", split=split)):
        for record in process_rpr_row(row):
            is_passing_random = random.random() < sample_rate
            if force_gpt4 and record["creator"] == "gpt-4":
                records.append(record)
            elif random.random() < sample_rate:
                records.append(record)
    return records


def process_pippa(sample_rate, max_length):
    records = []
    for row in tqdm(load_dataset("PygmalionAI/PIPPA", split="train")):
        if random.random() > sample_rate:
            continue
        context = row["bot_description"]
        char_name = row["bot_name"]
        messages = revert_flattening(row["conversation"])
        system_message = f"You are {char_name}. {context}"
        chat = [{"role": "system", "content": system_message}]
        chat.append({
            "role": "prompt",
            "content": messages[0]["message"]
        })
        for message in messages[1:]:
            role = "user" if message["is_human"] else "bot"
            chat.append({
                "role": role,
                "content": message["message"]
            })
        length = calc_max_length([{"messages": chat}])
        while length > max_length:
            chat = chat[:-2]
            length = calc_max_length([{"messages": chat}])
        records.append({
            "messages": chat,
            "char_name": char_name,
            "source": "pippa"
        })
    print("PIPPA count:", len(records))
    if records:
        print("PIPPA max length:", calc_max_length(records))
    return records


def main(config_path, output_dir):
    random.seed(42)
    with open(config_path) as r:
        config = json.load(r)
    records = []

    if "pippa" in config:
        pippa_records = process_pippa(**config["pippa"])
        records += pippa_records

    # RPR
    rp_records = []
    if "rpr_en" in config:
        rp_records += process_rpr("en", **config["rpr_en"])
    if "rpr_ru" in config:
        rp_records += process_rpr("ru", **config["rpr_ru"])
    print("RPR count:", len(rp_records))
    if rp_records:
        print("RPR max length:", calc_max_length(rp_records))
    records += rp_records

    # Positive conversations
    pos_records = []
    pos_config = config["pos"]
    for row in tqdm(load_dataset("IlyaGusev/chai_prize_positive_conversations", split="train")):
        if random.random() > pos_config["sample_rate"]:
            continue
        chat = revert_flattening(row["messages"])
        char_name = row["char_name"]
        chat.insert(0, {
            "role": "system",
            "content": "You are {char_name}."
        })
        pos_records.append({
            "messages": chat,
            "source": "pos_feedback"
        })
    print("From positive feedback count:", len(pos_records))
    if pos_records:
        print("Pos max length:", calc_max_length(pos_records))
    records += pos_records

    # GPTeacher
    gpteacher_records = []
    gpteacher_config = config["gpteacher"]
    for row in tqdm(load_dataset("AlekseyKorshuk/gpteacher-role-play-chatml", split="train")):
        if random.random() > gpteacher_config["sample_rate"]:
            continue
        chat = []
        for message in row["conversation"]:
            content = message["content"]
            role = message["role"]
            if role == "User":
                chat.append({
                    "role": "user",
                    "content": content
                })
            else:
                chat.append({
                    "role": "bot",
                    "content": content
                })
        random_number = random.random()
        if random_number < 0.3:
            chat[0]["role"] = "system"
        elif random_number < 0.2:
            chat[0]["role"] = "prompt"
        gpteacher_records.append({
            "messages": chat,
            "source": "gpteacher"
        })
    records += gpteacher_records
    print("GPTeacher count:", len(gpteacher_records))
    if gpteacher_records:
        print("GPTeacher max length:", calc_max_length(gpteacher_records))

    # Final processing
    cleaned_records = []
    for record in records:
        messages = record["messages"]
        roles = {m["role"] for m in messages}
        for role in roles:
            assert role in ("bot", "user", "system", "prompt"), role
        if not record["messages"]:
            continue
        cleaned_records.append(record)
    records = cleaned_records
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
