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
    has_empty_messages
)
from chai_prize.datasets.chai import (
    parse_chai_conversation,
    is_whitelisted_model,
    is_good_feedback
)


def calc_user_engagement(messages):
    response_length = [len(m["content"]) for m in messages if m["role"] == "user"]
    if len(response_length) <= 1:
        return 0.0
    response_length = response_length[1:]
    return median(response_length)


def calc_bot_questions(messages):
    return sum([int("?" in m["content"]) for m in messages if m["role"] == "bot"])


def bot_has_long_answers(chat, min_chars: int = 150):
    bot_messages = [m["content"] for m in chat if m["role"] == "bot"]
    if len(bot_messages) > 1:
        bot_messages = bot_messages[1:]
    result = min([len(m) for m in bot_messages])
    return result > min_chars


def bot_has_actions(chat, min_fraction: float = 0.85):
    bot_messages = [m["content"] for m in chat if m["role"] == "bot"]
    actions_count = sum([int("*" in m or '"' in m) for m in bot_messages])
    return actions_count >= int(len(bot_messages) * min_fraction)


def bot_has_questions(chat, min_fraction: float = 0.6):
    bot_messages = [m["content"] for m in chat if m["role"] == "bot"]
    num_questions = calc_bot_questions(chat)
    return num_questions >= int(len(bot_messages) * min_fraction)


def has_bad_ss(chat):
    ss = (
        " AI",
        "language model",
        "Chai"
    )
    for message in chat:
        content = message["content"]
        for s in ss:
            if s in content:
                return True
    return False


def add_ctrl_attributes(chat, row):
    counts = Counter()
    attributes = []
    context = chat[0]["content"]
    if bot_has_long_answers(chat):
        counts["verbose"] += 1
        attributes.append("Verbosity: high")
    else:
        attributes.append("Verbosity: low")
    if bot_has_actions(chat):
        counts["actions"] += 1
        attributes.append("Actions: many")
    else:
        attributes.append("Actions: few")

    role_play_score = row.get("role_play_score")
    if role_play_score is not None:
        attributes.append(f"Role play: {role_play_score}")

    consciousness_score = row.get("consciousness_score")
    if consciousness_score is not None:
        attributes.append(f"Consciousness: {consciousness_score}")

    context += "\n#### Controls:\n" + "\n".join(attributes) + "\n"
    chat[0]["content"] = context
    return counts


def process_rpr_row(row, add_ctrl: bool = False):
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
        current_chat = copy.deepcopy(chat)
        current_chat += dialogue["chat"]
        for message in current_chat:
            if message["role"] == "char":
                message["role"] = "bot"
            if message["role"] == "operator":
                message["role"] = "user"
        if add_ctrl:
            add_ctrl_attributes(current_chat, row)
        yield {
            "messages": current_chat,
            "creator": dialogue["model_name"],
            "char_name": row["name"].strip(),
            "source": "rpr"
        }


def process_rpr(
    split: str = "en",
    sample_rate: float = 1.0,
    force_gpt4: bool = True,
    dataset_name: str = "IlyaGusev/gpt_roleplay_realm",
    add_ctrl: bool = False,
    **kwargs
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split=split)):
        for record in process_rpr_row(row, add_ctrl=add_ctrl):
            if force_gpt4 and record["creator"] != "gpt-4":
                continue
            if random.random() < sample_rate:
                records.append(record)
    return records


def get_score(row, field):
    score = row.get(field + "_score", 0)
    score = score if score else 0
    return score


def process_chai(
    sample_rate: float = 1.0,
    dataset_name: str = "ChaiML/20231007_chai_prize_model_feedback_all",
    character_dataset_name: str = "ChaiML/seasonIII_chatAI_configurations",
    max_length: int = 20000,
    min_num_bot_questions: int = 0,
    add_ctrl: bool = False,
    min_messages: int = 4,
    only_thumbs_up: bool = False,
    only_good_feedback: bool = False,
    min_action_level: int = 0,
    min_user_engagement: int = 0,
    min_creativity: int = 0,
    min_action_heuristics_score: float = 0.0,
    min_user_engagement_heuristics_score: float = 0.0,
    only_whitelist: bool = False
):
    records = []
    ctrl_counts = Counter()
    characters = {row["bot_id"]: row for row in load_dataset(character_dataset_name, split="train")}
    for row in tqdm(load_dataset(dataset_name, split="train")):
        bot_id = row["bot_id"]

        if bot_id not in characters:
            continue

        if only_thumbs_up and not row["thumbs_up"]:
            continue

        text = row["conversation"]
        if "INST" in text or "START" in text:
            continue

        char_name = text.split(":")[0].strip()
        chat = list(parse_chai_conversation(text))
        chat = [{"role": m["role"], "content": m["content"]} for m in chat if not m["is_deleted"]]
        if not chat:
            continue

        if has_repetition(chat):
            continue

        if has_empty_messages(chat):
            continue

        if has_bad_ss(chat):
            continue

        if calc_user_engagement(chat) < min_user_engagement_heuristics_score:
            continue

        if calc_bot_questions(chat) < min_num_bot_questions:
            continue

        if len(chat) < min_messages:
            continue

        if not bot_has_actions(chat, min_action_heuristics_score):
            continue

        if only_whitelist and not is_whitelisted_model(row["model_name"]):
            continue

        if only_good_feedback and not is_good_feedback(row["feedback"]):
            continue

        if get_score(row, "action_level") < min_action_level:
            continue

        if get_score(row, "user_engagement") < min_user_engagement:
            continue

        if get_score(row, "creativity") < min_creativity:
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

        if not has_bot_message(chat):
            continue

        for message in chat:
            content = message["content"]
            content = content if message["role"] == "user" else clean_bot_message(content)
            message["content"] = content

        if add_ctrl:
            row_counts = add_ctrl_attributes(chat, row)
            ctrl_counts += row_counts

        records.append({
            "messages": chat,
            "char_name": char_name,
            "original_fields": row,
            "source": "chai"
        })

    if ctrl_counts:
        print("CTRL:", ctrl_counts)
    print("From Chai count:", len(records))
    if records:
        print("Chai max length:", calc_max_length(records))
    return records


def process_pippa(
    sample_rate: float = 1.0,
    max_length: int = 20000,
    dataset_name: str = "PygmalionAI/PIPPA",
    min_num_bot_questions: int = 0,
    use_random_roles: bool = False,
    add_ctrl: bool = False,
    min_messages: int = 4,
    min_action_level: int = 0,
    min_user_engagement: int = 0,
    min_creativity: int = 0,
    min_action_heuristics_score: float = 0.0,
    min_user_engagement_heuristics_score: float = 0.0
):
    records = []

    ctrl_counts = Counter()
    for row in tqdm(load_dataset(dataset_name, split="train")):
        context = row["bot_description"]
        char_name = row["bot_name"].strip()
        messages = revert_flattening(row["conversation"])
        if len(messages) < min_messages:
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

        if calc_user_engagement(chat) < min_user_engagement_heuristics_score:
            continue

        if calc_bot_questions(chat) < min_num_bot_questions:
            continue

        chat = shrink(chat, max_length)
        if not has_bot_message(chat):
            continue

        if use_random_roles and random.random() < 0.1:
            chat[0]["content"], chat[1]["content"] = chat[1]["content"], chat[0]["content"]

        if not bot_has_actions(chat, min_action_heuristics_score):
            continue

        if has_bad_ss(chat):
            continue

        if get_score(row, "action_level") < min_action_level:
            continue

        if get_score(row, "user_engagement") < min_user_engagement:
            continue

        if get_score(row, "creativity") < min_creativity:
            continue

        if random.random() > sample_rate:
            continue

        if add_ctrl:
            row_counts = add_ctrl_attributes(chat, row)
            ctrl_counts += row_counts

        records.append({
            "messages": chat,
            "char_name": char_name,
            "bot_id": row["bot_id"],
            "submission_timestamp": int(row["submission_timestamp"].timestamp()),
            "categories": row["categories"],
            "original_fields": row,
            "source": "pippa"
        })
    if ctrl_counts:
        print("CTRL:", ctrl_counts)
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
    add_ctrl: bool = False,
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
                if add_ctrl:
                    add_ctrl_attributes(current_chat, row)
                records.append({
                    "messages": current_chat,
                    "char_name": current_char_name.strip(),
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

    # Chai conversations
    if "chai" in config or "pos" in config:
        chai_records = process_chai(**config.get("chai", config.get("pos")))
        records += chai_records

    # GPTeacher
    if "gpteacher" in config:
        gpteacher_records = process_gpteacher(**config["gpteacher"])
        records += gpteacher_records

    # Final processing
    cleaned_records = []
    for record in tqdm(records):
        messages = record["messages"]
        roles = {m["role"] for m in messages}
        for role in roles:
            assert role in ("bot", "user", "system", "prompt"), role
        if not record["messages"]:
            continue
        if has_repetition(record["messages"]):
            continue
        if has_empty_messages(record["messages"]):
            continue
        cleaned_records.append(record)
    records = cleaned_records
    print("All count after cleaning:", len(records))

    for record in records:
        if "original_fields" not in record:
            continue
        record["original_fields"].pop("submission_timestamp", None)

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
