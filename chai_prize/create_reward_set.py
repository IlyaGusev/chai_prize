import os
import json
import copy
import random
from pathlib import Path
from collections import defaultdict, Counter

import fire
from datasets import load_dataset
from tqdm import tqdm

from chai_prize.util.data import (
    has_bot_message,
    calc_max_length,
    shrink,
    has_correct_roles,
    is_bad_chat,
    has_repetition,
    remove_trailing_user_messages,
    bot_has_wrong_language
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
        remove_trailing_user_messages(chat)
        if len(chat) < 5:
            continue

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


def process_arena(
    dataset_name: str = "lmsys/chatbot_arena_conversations",
):
    records = defaultdict(list)
    for row in tqdm(load_dataset(dataset_name, split="train")):
        model_a = row["model_a"]
        model_b = row["model_b"]
        model_a, model_a = sorted([model_a, model_b])
        conversation_a = row["conversation_a"]
        conversation_b = row["conversation_b"]
        conv1, conv2 = sorted([str(conversation_a), str(conversation_b)])
        qid = hash((conv1, conv2))
        records[(qid, model_a, model_b)].append(row)
    final_records = []
    for idx, (_, question_records) in enumerate(records.items()):
        cnt = Counter()
        for r in question_records:
            if r["winner"] in ("tie", "tie (bothbad)"):
                continue
            cnt[r[r["winner"]]] += 1
        if not cnt:
            continue
        winner_model = cnt.most_common(1)[0][0]
        record = question_records[0]
        conv_a = record["conversation_a"]
        conv_b = record["conversation_b"]
        for m in conv_a:
            if m["role"] == "assistant":
                m["role"] = "bot"
        for m in conv_b:
            if m["role"] == "assistant":
                m["role"] = "bot"
        assert winner_model in (record["model_a"], record["model_b"])
        chosen_messages = conv_a if record["model_a"] == winner_model else conv_b
        rejected_messages = conv_b if record["model_a"] == winner_model else conv_a
        char_name = "Arena bot {}".format(idx)
        chosen_messages.insert(0, {"role": "system", "content": "You are a helpful assistant"})
        chosen_messages.insert(1, {"role": "prompt", "content": ""})
        rejected_messages.insert(0, {"role": "system", "content": "You are a helpful assistant"})
        rejected_messages.insert(1, {"role": "prompt", "content": ""})
        final_records.append({
            "chosen_messages": chosen_messages,
            "rejected_messages": rejected_messages,
            "char_name": char_name
        })
    print("Arena count: {}".format(len(final_records)))
    return final_records


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
    modes = config["modes"]

    if "arena" in modes:
        final_records.extend(process_arena())

    if "deleted" in modes:
        for record in chai_records:
            chat = record["messages"]
            if sum([m.get("is_deleted", False) for m in chat]) == 0:
                continue
            if not record["original_fields"]["thumbs_up"]:
                continue

            deleted_messages = []
            prev_messages = []
            for message in chat:
                if message.get("is_deleted", False):
                    deleted_messages.append(message)
                    continue

                if len(deleted_messages) >= 1 and len(deleted_messages) <= 2:
                    for rejected_message in deleted_messages:
                        if prev_messages[-1]["role"] != "user":
                            continue
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

    if "thumbs_up" in modes:
        char_records = defaultdict(list)
        for record in chai_records:
            record["messages"] = [m for m in record["messages"] if not m.get("is_deleted", False)]
            char_records[record["bot_id"]].append(record)

        for _, records in char_records.items():
            positive_records = []
            negative_records = []
            for r in records:
                assert r["messages"][-1]["role"] == "bot", r["messages"][-1]["role"]
                is_thumbs_up = r["original_fields"]["thumbs_up"]
                is_repeating = has_repetition(r["messages"][-4:], prefix_length=30)
                has_wrong_language = bot_has_wrong_language(r["messages"])
                if has_wrong_language:
                    negative_records.append(r)
                    continue
                if is_repeating:
                    negative_records.append(r)
                    continue
                if is_thumbs_up:
                    positive_records.append(r)
                    continue
                negative_records.append(r)

            if not positive_records or not negative_records:
                continue
            random.shuffle(positive_records)
            random.shuffle(negative_records)
            char_name = positive_records[0]["char_name"]

            for pos_record, neg_record in zip(positive_records, negative_records):
                final_records.append({
                    "chosen_messages": pos_record["messages"],
                    "rejected_messages": neg_record["messages"],
                    "char_name": char_name
                })
            for pos_record in positive_records:
                messages = pos_record["messages"]
                if len(messages) >= 7 and random.random() < config.get("aug_swap_prob", 0.0):
                    neg_messages = copy.deepcopy(messages)
                    bot_message_indices = [i for i, m in enumerate(messages) if m["role"] == "bot"][:-1]
                    swap_index = random.choice(bot_message_indices)
                    neg_messages[-1], neg_messages[swap_index] = messages[swap_index], messages[-1]
                    final_records.append({
                        "chosen_messages": messages[:],
                        "rejected_messages": neg_messages,
                        "char_name": char_name,
                        "aug": "swap"
                    })
                if random.random() < config.get("aug_empty_prob", 0.0):
                    neg_messages = copy.deepcopy(messages)
                    neg_messages[-1]["content"] = ""
                    final_records.append({
                        "chosen_messages": messages[:],
                        "rejected_messages": neg_messages,
                        "char_name": char_name,
                        "aug": "empty"
                    })
                if random.random() < config.get("aug_shuffle_prob", 0.0):
                    neg_messages = copy.deepcopy(messages)
                    content = neg_messages[-1]["content"]
                    words = content.split()
                    random.shuffle(words)
                    new_content = " ".join(words)
                    if new_content != content:
                        neg_messages[-1]["content"] = new_content
                    final_records.append({
                        "chosen_messages": messages[:],
                        "rejected_messages": neg_messages,
                        "char_name": char_name,
                        "aug": "shuffle"
                    })

    if "thumbs_up_pointwise" in modes:
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
    print("All count after cleaning:", len(final_records))

    char_names = list({r["char_name"] for r in final_records})
    random.shuffle(char_names)
    border = int(0.95 * len(char_names))
    train_chars = set(char_names[:border])
    val_chars = set(char_names[border:])
    train_records = [r for r in final_records if r["char_name"] in train_chars]
    random.shuffle(train_records)
    val_records = [r for r in final_records if r["char_name"] in val_chars]
    random.shuffle(val_records)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, "train.jsonl"), "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(os.path.join(output_dir, "val.jsonl"), "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)