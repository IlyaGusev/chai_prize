import os
import json
import random
from pathlib import Path
from statistics import median

import fire
from datasets import load_dataset
from tqdm import tqdm


DEFAULT_SYSTEM_TEMPLATE = "{char_name}'s Persona: {content}"


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
    if len(response_length) <= 1:
        return 0.0
    response_length = response_length[1:]
    return median(response_length)


def calc_bot_questions(messages):
    return sum([int("?" in m["content"]) for m in messages if m["role"] == "bot"])


def shrink(chat, max_length):
    length = calc_max_length([{"messages": chat}])
    while length > max_length:
        chat = chat[:-2]
        length = calc_max_length([{"messages": chat}])
    return chat


def has_bot_message(chat):
    roles = {m["role"] for m in chat}
    return "bot" in roles


def clean_bot_message(text):
    text = " ".join(text.split())
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    return text


def has_repetition(chat):
    bot_messages = [m["content"] for m in chat if m["role"] == "bot"]
    uniq_bot_messages = set(bot_messages)
    return len(uniq_bot_messages) < len(bot_messages)


def process_rpr_row(row):
    name = row["name"]
    greeting = row["greeting"]
    context = row["context"]
    example_dialogue = row["example_dialogue"]

    chat = [{
        "role": "system",
        "content": context
    }]

    mapping = {
        "user": "User",
        "char": name
    }
    prompt = [f'{mapping[m["role"]]}: {m["content"]}' for m in example_dialogue]
    prompt = "\n".join(prompt)
    chat.append({
        "role": "prompt",
        "content": prompt
    })
    chat.append({
        "role": "bot",
        "content": greeting
    })

    for dialogue in row["dialogues"]:
        chat += dialogue["chat"]
        for message in chat:
            if message["role"] == "char":
                message["role"] = "bot"
            if message["role"] == "operator":
                message["role"] = "user"

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
    dataset_name: str = "IlyaGusev/gpt_roleplay_realm",
    **kwargs
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
    dataset_name: str = "IlyaGusev/chai_prize_positive_conversations",
    min_user_engagement: float = 50.0,
    max_length: int = 6000,
    min_num_bot_questions: int = 0,
    min_score: int = 0,
    min_user_engagement_score: int = 0,
    min_role_play_score: int = 0,
    promote_nsfw: bool = False,
    **kwargs
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split="train")):
        if random.random() > sample_rate:
            continue
        chat = revert_flattening(row["messages"])

        engagement = calc_user_engagement(chat)
        if engagement < min_user_engagement:
            continue
        num_bot_questions = calc_bot_questions(chat)
        if num_bot_questions < min_num_bot_questions:
            continue

        char_name = row["char_name"]
        if chat[0]["role"] != "system":
            chat.insert(0, {
                "role": "system",
                "content": ""
            })
        else:
            chat[0]["content"] = ""

        if chat[1]["role"] == "prompt" and chat[2]["role"] == "user":
            chat[1]["role"] = "bot"
            chat.insert(1, {
                "role": "prompt",
                "content": ""
            })
        chat = shrink(chat, max_length)
        if not has_bot_message(chat):
            continue

        if "role_play_score" in row:
            score = row["role_play_score"] + row["consciousness_score"] + row["user_engagement_score"]
            if promote_nsfw:
                score += row["nsfw_score"] // 2
            if score < min_score:
                continue
            if row["user_engagement_score"] < min_user_engagement_score:
                continue
            if row["role_play_score"] < min_role_play_score:
                continue

        if has_repetition(chat):
            continue

        records.append({
            "messages": chat,
            "char_name": char_name,
            "conversation_id": row["conversation_id"],
            "source": "pos_feedback"
        })
    print("From positive feedback count:", len(records))
    if records:
        print("Pos max length:", calc_max_length(records))
    return records


def process_pippa(
    sample_rate: float = 1.0,
    max_length: int = 20000,
    min_user_engagement: float = 50.0,
    dataset_name: str = "PygmalionAI/PIPPA",
    min_num_bot_questions: int = 0,
    min_score: int = 0,
    promote_nsfw: bool = False,
    min_user_engagement_score: int = 0,
    min_role_play_score: int = 0,
    use_random_roles: bool = False,
    **kwargs
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split="train")):
        if random.random() > sample_rate:
            continue
        context = row["bot_description"]
        char_name = row["bot_name"]
        messages = revert_flattening(row["conversation"])
        if len(messages) <= 3:
            continue

        chat = [{"role": "system", "content": context}]

        prompt = row["bot_definitions"]
        prompt = prompt.split("END_OF_DIALOG")[0]
        prompt = prompt.replace("{{user}}", "User")
        prompt = prompt.replace("{{user}", "User")
        prompt = prompt.replace("{{u01}}", "User")
        for i in range(20):
            prompt = prompt.replace("{{random_user_" + str(i) + "}}", "User")
        prompt = prompt.replace("{{char}}", char_name)
        prompt = prompt.strip()
        if use_random_roles and random.random() < 0.1:
            prompt = ""
        chat.append({
            "role": "prompt",
            "content": prompt
        })

        if use_random_roles and random.random() < 0.1:
            chat[0]["content"] = prompt
            chat[1]["content"] = context

        for message in messages:
            role = "user" if message["is_human"] else "bot"
            content = message["message"]
            content = content if role == "user" else clean_bot_message(content)
            if content.startswith(char_name + ":"):
                content = content[len(char_name) + 1:].strip()
            chat.append({
                "role": role,
                "content": content,
            })

        if sum(["{{user}}" in message["message"] for message in messages[1:]]) > 0:
            continue

        engagement = calc_user_engagement(chat)
        if engagement < min_user_engagement:
            continue

        num_bot_questions = calc_bot_questions(chat)
        if num_bot_questions < min_num_bot_questions:
            continue

        chat = shrink(chat, max_length)
        if not has_bot_message(chat):
            continue

        if "role_play_score" in row:
            score = row["role_play_score"] + row["consciousness_score"] + row["user_engagement_score"]
            if promote_nsfw:
                score += row["nsfw_score"] // 2
            if score < min_score:
                continue
            if row["user_engagement_score"] < min_user_engagement_score:
                continue
            if row["role_play_score"] < min_role_play_score:
                continue

        records.append({
            "messages": chat,
            "char_name": char_name,
            "bot_id": row["bot_id"],
            "submission_timestamp": int(row["submission_timestamp"].timestamp()),
            "categories": row["categories"],
            "source": "pippa"
        })
    print("PIPPA count:", len(records))
    if records:
        print("PIPPA max length:", calc_max_length(records))
    return records


def process_gpteacher(
    sample_rate: float = 1.0,
    dataset_name: str = "AlekseyKorshuk/gpteacher-role-play-chatml",
    **kwargs
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split="train")):
        if random.random() > sample_rate:
            continue
        chat = []
        for message in row["conversation"]:
            content = message["content"]
            role = message["role"]
            chat.append({
                "role": "user" if role == "User" else "bot",
                "content": content
            })
        chat[0]["role"] = "system"
        chat.insert(1, {
            "role": "prompt",
            "content": ""
        })
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
    max_length: int = 20000,
    **kwargs
):
    current_conversation_id = None
    current_message_id = None
    current_chat = []
    current_char_name = None
    records = []
    for row in load_dataset(dataset_name, split="train"):
        message = row["message"]
        message_type = row["message_type"]
        conversation_id = row["conversation_id"]
        message_id = row["message_id"]
        if current_conversation_id != conversation_id:
            if current_chat:
                current_chat = shrink(current_chat, max_length)
                records.append({
                    "messages": current_chat,
                    "char_name": current_char_name,
                    "source": "limarp"
                })
            current_chat = []
            current_conversation_id = conversation_id
        if current_chat:
            assert message_id == current_message_id + 1
        current_message_id = message_id
        if message_type == "instruction":
            current_char_name = message.split("'s")[0]
            current_chat.append({
                "role": "system",
                "content": message
            })
            current_chat.append({
                "role": "prompt",
                "content": ""
            })
        else:
            content = ":".join(message.split(":")[1:])
            assert message_type in ("input", "output")
            role = "user" if message_type == "input" else "bot"
            current_chat.append({
                "role": role,
                "content": " ".join(content.strip().split())
            })

    final_records = []
    for record in records:
        if not has_bot_message(record["messages"]):
            continue
        if random.random() < sample_rate:
            final_records.append(record)
    records = final_records

    print("LIMArp count:", len(records))
    if records:
        print("LIMArp max length:", calc_max_length(records))
    return records


def process_bluemoon(
    sample_rate: float = 1.0,
    dataset_name: str = "seank0602/bluemoon_fandom_rp",
    max_length: int = 20000,
    **kwargs
):
    records = []
    for row in load_dataset(dataset_name, split="train"):
        conversation_id = row["id"]
        chat = row["conversations"]
        for message in chat:
            role = message.pop("from")
            message["role"] = "user" if role == "human" else "bot"
            message["content"] = message.pop("value")
        system_messages = [{
            "role": "system",
            "content": ""
        }, {
            "role": "prompt",
            "content": ""
        }]
        chat = system_messages + chat
        chat = shrink(chat, max_length)
        if not has_bot_message(chat):
            continue
        records.append({
            "messages": chat,
            "char_name": "Character",
            "source": "bluemoon"
        })
    print("Bluemoon count:", len(records))
    if records:
        print("Bluemoon max length:", calc_max_length(records))
    return records


def process_ao3(
    sample_rate: float = 1.0,
    dataset_name: str = "ebony59/AO3_fandom_chai",
    max_length: int = 20000,
    **kwargs
):
    records = []
    for row in load_dataset(dataset_name, split="train"):
        conversations = row["conversations"]
        chat = [{
            "role": "system",
            "content": row["personalities"]
        }, {
            "role": "prompt",
            "content": row["prompt"]
        }]
        char1 = row["character_1"]
        char2 = row["character_2"]
        characters = {message["role"] for message in conversations}
        if len(characters) > 2 or char1 not in characters or char2 not in characters:
            continue
        for message in conversations:
            chat.append({
                "role": "user" if message["role"] == char2 else "bot",
                "content": message["content"].strip()
            })
        chat = shrink(chat, max_length)
        if not has_bot_message(chat):
            continue
        records.append({
            "messages": chat,
            "char_name": char1,
            "source": "ao3"
        })
    print("AO3 count:", len(records))
    if records:
        print("AO3 max length:", calc_max_length(records))
    return records


def main(config_path, output_dir):
    random.seed(42)
    with open(config_path) as r:
        config = json.load(r)
    records = []

    # AO3
    if "ao3" in config:
        ao3_records = process_ao3(**config["ao3"])
        records += ao3_records

    # Bluemoon
    if "bluemoon" in config:
        bluemoon_records = process_bluemoon(**config["bluemoon"])
        records += bluemoon_records

    # LIMA RP
    if "limarp" in config:
        limarp_records = process_limarp(**config["limarp"])
        records += limarp_records

    # PIPPA
    if "pippa" in config:
        pippa_records = process_pippa(**config["pippa"])
        records += pippa_records

    # RPR
    rp_records = []
    if "rpr_en" in config:
        rp_records += process_rpr("en", **config["rpr_en"])
    if "rpr_ru" in config:
        rp_records += process_rpr("ru", **config["rpr_ru"])
    if rp_records:
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
        if has_repetition(record["messages"]):
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
