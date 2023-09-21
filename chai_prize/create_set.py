import os
import json
import random
from pathlib import Path

import fire
from datasets import load_dataset
from tqdm import tqdm


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


def calc_user_engagement(messages):
    response_length = [len(m["content"]) for m in messages if m["role"] == "user"]
    if len(response_length) == 0:
        return 0.0
    return sum(response_length) / len(response_length)


def shrink(chat, max_length):
    length = calc_max_length([{"messages": chat}])
    while length > max_length:
        chat = chat[:-2]
        length = calc_max_length([{"messages": chat}])
    return chat


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
    force_gpt4: bool = True,
    dataset_name: str = "IlyaGusev/gpt_roleplay_realm"
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split=split)):
        for record in process_rpr_row(row):
            if force_gpt4 and record["creator"] == "gpt-4":
                records.append(record)
            elif random.random() < sample_rate:
                records.append(record)
    return records


def process_pos(
    sample_rate: float = 1.0,
    dataset_name: str = "IlyaGusev/chai_prize_positive_conversations"
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split="train")):
        if random.random() > sample_rate:
            continue
        chat = revert_flattening(row["messages"])
        char_name = row["char_name"]
        chat.insert(0, {
            "role": "system",
            "content": "You are {char_name}."
        })
        records.append({
            "messages": chat,
            "char_name": char_name,
            "source": "pos_feedback"
        })
    print("From positive feedback count:", len(records))
    if records:
        print("Pos max length:", calc_max_length(records))
    return records


def process_pippa(
    sample_rate: float = 1.0,
    max_length: int = 20000,
    min_user_engagement: float = 100.0,
    dataset_name: str = "PygmalionAI/PIPPA"
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split="train")):
        if random.random() > sample_rate:
            continue
        context = row["bot_description"]
        char_name = row["bot_name"]
        messages = revert_flattening(row["conversation"])
        system_message = f"You are {char_name}. {context}"
        chat = [{"role": "system", "content": system_message}]
        role = "prompt"
        if random.random() < 0.2:
            role = "bot"
        chat.append({
            "role": role,
            "content": messages[0]["message"]
        })
        for message in messages[1:]:
            role = "user" if message["is_human"] else "bot"
            chat.append({
                "role": role,
                "content": message["message"]
            })
        engagement = calc_user_engagement(chat[3:])
        if engagement < min_user_engagement:
            continue

        chat = shrink(chat, max_length)
        records.append({
            "messages": chat,
            "char_name": char_name,
            "source": "pippa"
        })
    print("PIPPA count:", len(records))
    if records:
        print("PIPPA max length:", calc_max_length(records))
    return records


def process_gpteacher(
    sample_rate: float = 1.0,
    dataset_name: str = "AlekseyKorshuk/gpteacher-role-play-chatml"
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split="train")):
        if random.random() > sample_rate:
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
        records.append({
            "messages": chat,
            "char_name": "bot",
            "source": "gpteacher"
        })
    print("GPTeacher count:", len(records))
    if records:
        print("GPTeacher max length:", calc_max_length(records))
    return records


def process_limarp(
    sample_rate: float = 1.0,
    dataset_name: str = "TearGosling/limarp_standardized",
    max_length: int = 20000
):
    current_conversation_id = None
    current_message_id = None
    current_chat = []
    records = []
    for row in load_dataset(dataset_name, split="train"):
        message = row["message"]
        message_type = row["message_type"]
        conversation_id = row["conversation_id"]
        message_id = row["message_id"]
        if current_conversation_id != conversation_id:
            if current_chat:
                char_name = current_chat[0]["content"].split("'s")[0]
                current_chat = shrink(current_chat, max_length)
                records.append({
                    "messages": current_chat,
                    "char_name": char_name,
                    "source": "limarp"
                })
            current_chat = []
            current_conversation_id = conversation_id
        if current_chat:
            assert message_id == current_message_id + 1
        current_message_id = message_id
        if message_type == "instruction":
            role = "system"
            if random.random() < 0.2:
                role = "prompt"
            current_chat.append({
                "role": role,
                "content": message
            })
        else:
            content = ":".join(message.split(":")[1:])
            assert message_type in ("input", "output")
            role = "user" if message_type == "input" else "bot"
            current_chat.append({
                "role": role,
                "content": content
            })

    final_records = []
    for record in records:
        if random.random() < sample_rate:
            final_records.append(record)
    records = final_records

    print("LIMArp count:", len(records))
    if records:
        print("LIMArp max length:", calc_max_length(records))
    return records


def main(config_path, output_dir):
    random.seed(42)
    with open(config_path) as r:
        config = json.load(r)
    records = []

    if "limarp" in config:
        limarp_records = process_limarp(**config["limarp"])
        records += limarp_records

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
    if "pos" in config:
        pos_records = process_pos(**config["pos"])
        records += pos_records

    # GPTeacher
    if "gpteacher" in config:
        gpteacher_records = process_gpteacher(**config["gpteacher"])
        records += gpteacher_records

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
