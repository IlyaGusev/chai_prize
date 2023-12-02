import os
import json
import copy
import random
from typing import Dict, Optional
from collections import Counter
from pathlib import Path
from statistics import median

import fire
from datasets import load_dataset
from tqdm import tqdm

from chai_prize.util.data import (
    is_bad_chat,
    revert_flattening,
    clean_bot_message,
    calc_max_length,
    shrink,
    is_single_character,
    is_not_english,
    bot_has_wrong_language,
    remove_trailing_user_messages,
    has_actions,
    has_ai_ss
)
from chai_prize.datasets.chai import (
    parse_chai_conversation,
    is_whitelisted_model,
    is_good_feedback
)


ASSISTANT_CHAR_NAME = "Assistant"
ASSISTANT_SYSTEM_MESSAGE = "The assistant gives helpful, detailed, and polite answers to the human's questions."
ASSISTANT_GREETING = "Hello!"


def calc_user_engagement(messages):
    response_length = [len(m["content"]) for m in messages if m["role"] == "user"]
    if len(response_length) <= 1:
        return 0.0
    response_length = response_length[1:]
    return median(response_length)


def calc_bot_engagement(messages):
    response_length = [len(m["content"]) for m in messages if m["role"] == "bot"]
    if len(response_length) <= 1:
        return 0.0
    response_length = response_length[1:]
    return median(response_length)


def bot_has_long_answers(chat, min_chars: int = 150):
    bot_messages = [m["content"] for m in chat if m["role"] == "bot"]
    if len(bot_messages) > 1:
        bot_messages = bot_messages[1:]
    result = min([len(m) for m in bot_messages])
    return result > min_chars


def bot_has_actions(chat, min_fraction: float = 0.85, min_action_length: int = 7):
    bot_messages = [m["content"] for m in chat if m["role"] == "bot"]
    count_action_messages = sum([has_actions(message, min_action_length) for message in bot_messages])
    return count_action_messages >= int(len(bot_messages) * min_fraction)


DEFAULT_CONTROLS = {"verbosity", "actions", "creativity", "capriciousness", "fragility"}


def add_ctrl_attributes(chat, row, controls=DEFAULT_CONTROLS):
    counts = Counter()
    attributes = []
    context = chat[0]["content"]

    if "verbosity" in controls:
        if bot_has_long_answers(chat):
            counts["verbose"] += 1
            attributes.append("Verbosity: high")
        else:
            attributes.append("Verbosity: low")

    if "actions" in controls:
        if "action_level_score" in row:
            action_level_score = get_score(row, "action_level")
            if action_level_score >= 7:
                counts["actions"] += 1
                attributes.append("Actions: many")
            else:
                attributes.append("Actions: few")
        else:
            if bot_has_actions(chat):
                counts["actions"] += 1
                attributes.append("Actions: many")
            else:
                attributes.append("Actions: few")

    if "creativity" in controls and "creativity_score" in row:
        creativity_score = get_score(row, "creativity")
        if creativity_score >= 5:
            counts["creativity"] += 1
            attributes.append("Creativity: high")
        else:
            attributes.append("Creativity: low")

    if "capriciousness" in controls and "capriciousness_score" in row:
        capriciousness_score = get_score(row, "capriciousness")
        if capriciousness_score >= 4:
            counts["capriciousness"] += 1
            attributes.append("Capriciousness: high")
        else:
            attributes.append("Capriciousness: low")

    if "fragility" in controls and "fragility_score" in row:
        fragility_score = get_score(row, "fragility")
        if fragility_score >= 4:
            counts["fragility"] += 1
            attributes.append("Fragility: high")
        else:
            attributes.append("Fragility: low")

    context += "\n#### Controls:\n" + "\n".join(attributes)
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
    add_ctrl: bool = False,
    min_messages: int = 4,
    only_thumbs_up: bool = False,
    only_good_feedback: bool = False,
    min_scores: Optional[Dict[str, int]] = None,
    min_action_heuristics_score: float = 0.0,
    min_user_engagement_heuristics_score: float = 0.0,
    min_bot_engagement_heuristics_score: float = 0.0,
    only_whitelist: bool = False,
    boost_not_english: bool = False,
    only_same_language: bool = False,
    exclude_last_message: bool = False,
    max_char_chats: Optional[int] = None,
    min_action_length: int = 7
):
    records = []
    ctrl_counts = Counter()
    char_counts = Counter()
    characters = {row["bot_id"]: row for row in load_dataset(character_dataset_name, split="train")}
    for row in tqdm(load_dataset(dataset_name, split="train")):
        # Join
        bot_id = row["bot_id"]
        if bot_id not in characters:
            continue
        character = characters[bot_id]

        # Parse conversations
        text = row["conversation"]
        if "INST" in text or "START" in text:
            continue

        char_name = text.split(":")[0].strip()
        chat = list(parse_chai_conversation(text))
        chat = [{"role": m["role"], "content": m["content"]} for m in chat if not m["is_deleted"]]
        remove_trailing_user_messages(chat)
        if not chat:
            continue
        if exclude_last_message:
            chat.pop()
        remove_trailing_user_messages(chat)
        if len(chat) < min_messages:
            continue

        for message in chat:
            content = message["content"]
            content = content if message["role"] == "user" else clean_bot_message(content)
            message["content"] = content

        memory = character["memory"]
        memory = memory if memory else ""
        memory = memory.strip()

        prompt = character["prompt"]
        prompt = prompt if prompt else ""
        prompt = prompt.strip()

        system_chat = [{"role": "system", "content": memory}, {"role": "prompt", "content": prompt}]
        chat = system_chat + chat
        chat = shrink(chat, max_length)

        # Filter
        if is_bad_chat(chat):
            continue

        if only_thumbs_up and not row["thumbs_up"]:
            continue

        if only_whitelist and not is_whitelisted_model(row["model_name"]):
            continue

        if only_same_language and bot_has_wrong_language(chat):
            continue

        if not boost_not_english or not is_not_english(chat):
            if is_single_character(char_name):
                if not bot_has_actions(chat, min_action_heuristics_score, min_action_length):
                    continue
                if calc_user_engagement(chat) < min_user_engagement_heuristics_score:
                    continue
                if calc_bot_engagement(chat) < min_bot_engagement_heuristics_score:
                    continue
            if only_good_feedback and not is_good_feedback(row["feedback"]):
                continue

        if min_scores:
            is_bad = False
            for key, min_score in min_scores.items():
                if get_score(row, key) < min_score:
                    is_bad = True
            if is_bad:
                continue

        if random.random() > sample_rate:
            continue

        char_counts[char_name] += 1
        if max_char_chats is not None:
            if char_counts[char_name] > max_char_chats:
                continue

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
    add_ctrl: bool = False,
    min_messages: int = 4,
    min_scores: Optional[Dict[str, int]] = None,
    min_action_heuristics_score: float = 0.0,
    min_user_engagement_heuristics_score: float = 0.0,
    min_bot_engagement_heuristics_score: float = 0.0,
    boost_not_english: bool = False,
    only_same_language: bool = False,
    max_char_chats: Optional[int] = None,
    min_action_length: int = 7
):
    records = []

    ctrl_counts = Counter()
    char_counts = Counter()
    fingerprints = set()
    for row in tqdm(load_dataset(dataset_name, split="train")):
        # Parse
        context = row["bot_description"]
        char_name = row["bot_name"].strip()
        messages = revert_flattening(row["conversation"])
        if len(messages) < min_messages:
            continue

        fingerprint = "\n".join([m["message"] for m in messages[:4]])
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)

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
        chat.append({"role": "prompt", "content": prompt})

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

        chat = shrink(chat, max_length)

        # Filter
        if is_bad_chat(chat):
            continue

        if only_same_language and bot_has_wrong_language(chat):
            continue

        if not boost_not_english or not is_not_english(chat):
            if is_single_character(char_name):
                if not bot_has_actions(chat, min_action_heuristics_score, min_action_length):
                    continue
                if calc_user_engagement(chat) < min_user_engagement_heuristics_score:
                    continue
                if calc_bot_engagement(chat) < min_bot_engagement_heuristics_score:
                    continue

        if min_scores:
            is_bad = False
            for key, min_score in min_scores.items():
                if get_score(row, key) < min_score:
                    is_bad = True
            if is_bad:
                continue

        if random.random() > sample_rate:
            continue

        char_counts[char_name] += 1
        if max_char_chats is not None:
            if char_counts[char_name] > max_char_chats:
                continue

        if add_ctrl:
            row_counts = add_ctrl_attributes(chat, row)
            ctrl_counts += row_counts

        timestamp = int(row.pop("submission_timestamp").timestamp())
        records.append({
            "messages": chat,
            "char_name": char_name,
            "bot_id": row["bot_id"],
            "submission_timestamp": timestamp,
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
        if is_bad_chat(chat):
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
        if random.random() > sample_rate:
            continue
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
        if is_bad_chat(chat):
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


def process_instruct_gpt4(
    sample_rate: float = 1.0,
    dataset_name: str = "lksy/ru_instruct_gpt4",
    max_length: int = 20000,
    rm_linebreaks: bool = False,
    **kwargs
):
    records = []
    for row in load_dataset(dataset_name, split="train"):
        message = row["instruction"]
        if row["input"]:
            message += "\nДано: " + row["input"]
        output = row["full_output"]
        if not output:
            continue
        if has_ai_ss([{"content": output}]):
            continue
        if rm_linebreaks:
            output = " ".join(output.split())
        chat = [{
            "role": "system",
            "content": ASSISTANT_SYSTEM_MESSAGE
        }, {
            "role": "prompt",
            "content": ""
        }, {
            "role": "bot",
            "content": ASSISTANT_GREETING
        }]
        chat.append({"role": "user", "content": message})
        chat.append({"role": "bot", "content": output})
        chat = shrink(chat, max_length)
        if is_bad_chat(chat):
            continue
        if random.random() > sample_rate:
            continue
        records.append({
            "messages": chat,
            "char_name": ASSISTANT_CHAR_NAME,
            "source": "gpt4"
        })
    print("GPT4 count:", len(records))
    if records:
        print("GPT4 max length:", calc_max_length(records))
    return records


def process_saiga(
    sample_rate: float = 1.0,
    dataset_name: str = "IlyaGusev/ru_turbo_saiga",
    max_length: int = 20000,
    rm_linebreaks: bool = False,
    **kwargs
):
    records = []
    for row in load_dataset(dataset_name, split="train"):
        messages = revert_flattening(row["messages"])
        if has_ai_ss(messages):
            continue
        if row["model_name"] != "gpt-4":
            continue
        chat = [{
            "role": "system",
            "content": ASSISTANT_SYSTEM_MESSAGE
        }, {
            "role": "prompt",
            "content": ""
        }, {
            "role": "bot",
            "content": ASSISTANT_GREETING
        }]
        if rm_linebreaks:
            for message in messages:
                if message["role"] == "bot":
                    message["content"] = " ".join(message["content"].split())

        chat += messages
        chat = shrink(chat, max_length)
        if is_bad_chat(chat):
            continue
        if random.random() > sample_rate:
            continue
        records.append({
            "messages": chat,
            "char_name": ASSISTANT_CHAR_NAME,
            "source": "saiga"
        })
    print("Saiga count:", len(records))
    if records:
        print("Saiga max length:", calc_max_length(records))
    return records


def process_oasst(
    sample_rate: float = 1.0,
    dataset_name: str = "IlyaGusev/oasst1_ru_main_branch",
    max_length: int = 20000,
    rm_linebreaks: bool = False,
    **kwargs
):
    oasst_records = []
    for row in tqdm(load_dataset(dataset_name, split="train")):
        messages = revert_flattening(row["messages"])
        if not messages:
            continue
        chat = [{
            "role": "system",
            "content": ASSISTANT_SYSTEM_MESSAGE
        }, {
            "role": "prompt",
            "content": ""
        }, {
            "role": "bot",
            "content": ASSISTANT_GREETING
        }]
        if rm_linebreaks:
            for message in messages:
                if message["role"] == "bot":
                    message["content"] = " ".join(message["content"].split())

        chat += messages
        chat = shrink(chat, max_length)
        if is_bad_chat(chat):
            continue
        if random.random() > sample_rate:
            continue
        oasst_records.append({
            "messages": chat,
            "char_name": ASSISTANT_CHAR_NAME,
            "source": "oasst"
        })
    print("OASST count:", len(oasst_records))
    if oasst_records:
        print("OASST max length:", calc_max_length(oasst_records))
    return oasst_records


FRP1_START = "Generate the next reply in this role-play chat as "


def process_freedomrp(
    sample_rate: float = 1.0,
    dataset_name: str = "openerotica/freedom-rp",
    max_length: int = 20000,
    min_messages: int = 4,
    min_action_heuristics_score: float = 0.0,
    min_user_engagement_heuristics_score: float = 0.0,
    min_bot_engagement_heuristics_score: float = 0.0,
    boost_not_english: bool = False,
    only_same_language: bool = False,
    min_action_length: int = 7,
    **kwargs
):
    records = []
    for row in load_dataset(dataset_name, split="train"):
        conversation = row["conversations"]
        if conversation[0]["from"] != "system":
            continue
        initial_system_message = conversation[0]["value"]
        if initial_system_message.startswith(FRP1_START):
            m = initial_system_message[len(FRP1_START):]
            char_name = m.split(":")[0]
            description = ":".join(m.split(":")[1:]).strip()
        else:
            continue
        chat = [{
            "role": "system",
            "content": description
        }, {
            "role": "prompt",
            "content": ""
        }]
        conversation = conversation[1:]
        if len(conversation) < min_messages:
            continue
        mapping = {
            "system": "system",
            "gpt": "bot",
            "human": "user"
        }
        for m in conversation:
            chat.append({
                "role": mapping[m["from"]],
                "content": m["value"]
            })
        chat = shrink(chat, max_length)
        if is_bad_chat(chat):
            continue

        if only_same_language and bot_has_wrong_language(chat):
            continue
        if not boost_not_english or not is_not_english(chat):
            if is_single_character(char_name):
                if not bot_has_actions(chat, min_action_heuristics_score, min_action_length):
                    continue
                if calc_user_engagement(chat) < min_user_engagement_heuristics_score:
                    continue
                if calc_bot_engagement(chat) < min_bot_engagement_heuristics_score:
                    continue
        if random.random() > sample_rate:
            continue

        records.append({
            "messages": chat,
            "char_name": char_name,
            "source": "freedom_rp"
        })
    print("Freedom RP count:", len(records))
    if records:
        print("Freedom RP max length:", calc_max_length(records))
    return records


def main(config_path, output_dir):
    random.seed(42)
    with open(config_path) as r:
        config = json.load(r)
    records = []

    # AO3
    if "ao3" in config:
        records += process_ao3(**config["ao3"])

    # Bluemoon
    if "bluemoon" in config:
        records += process_bluemoon(**config["bluemoon"])

    # LIMA RP
    if "limarp" in config:
        records += process_limarp(**config["limarp"])

    # PIPPA
    if "pippa" in config:
        records += process_pippa(**config["pippa"])

    # FreedomRP
    if "freedomrp" in config:
        records += process_freedomrp(**config["freedomrp"])

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
        records += process_chai(**config.get("chai", config.get("pos")))

    # GPTeacher
    if "gpteacher" in config:
        records += process_gpteacher(**config["gpteacher"])

    # Saiga
    if "ru_instruct_gpt4" in config:
        records += process_instruct_gpt4(**config["ru_instruct_gpt4"])

    if "saiga" in config:
        records += process_saiga(**config["saiga"])

    if "oasst" in config:
        records += process_oasst(**config["oasst"])

    # Final processing
    records = [r for r in records if not is_bad_chat(r["messages"])]

    print("Befre undup:", len(records))
    records = {
        (r["char_name"], [m for m in r["messages"] if m["role"] == "user"][0]["content"][:30]):
        r for r in records
    }
    records = list(records.values())

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
