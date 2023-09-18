import json
import sys
import random
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


def build_char_system_messages(char):
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
    if random.random() < 0.1:
        role = "prompt"
    chat.insert(0, {
        "role": "system",
        "content": full_context
    })
    return chat


def main(train_path, val_path):
    random.seed(42)
    records = []

    rp_records = []
    for row in tqdm(load_dataset("IlyaGusev/gpt_roleplay_realm", split="en")):
        for dialogue in row["dialogues"]:
            if dialogue["model_name"] != "gpt-4" and random.random() > 0.7:
                continue
            chat = dialogue["chat"]
            for message in chat:
                if message["role"] == "char":
                    message["role"] = "bot"
                if message["role"] == "operator":
                    message["role"] = "user"

            system_messages = build_char_system_messages(row)
            chat = system_messages + chat
            rp_records.append({
                "messages": chat,
                "source": "roleplay"
            })
    print("RPR count:", len(rp_records))
    print("RPR max length:", calc_max_length(rp_records))
    records += rp_records

    gpteacher_records = []
    for row in tqdm(load_dataset("AlekseyKorshuk/gpteacher-role-play-chatml", split="train")):
        assert len(row["conversation"]) == 2
        if random.random() > 0.2:
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
    print("GPTeacher max length:", calc_max_length(gpteacher_records))

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
    with open(train_path, "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(val_path, "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
