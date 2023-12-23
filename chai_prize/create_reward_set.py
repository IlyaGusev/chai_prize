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
    has_repetition,
    bot_has_wrong_language,
    has_actions,
    remove_actions
)
from chai_prize.create_set import process_chai


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
            char_records[record["char_name"]].append(record)

        for _, records in char_records.items():
            positive_records = []
            negative_records = []
            for r in records:
                assert r["messages"][-1]["role"] == "bot", r["messages"][-1]["role"]
                last_message = r["messages"][-1]["content"]
                orig_fields = r["original_fields"]
                is_thumbs_up = orig_fields.get("thumbs_up", orig_fields.get("labels"))
                is_repeating = has_repetition(r["messages"][-4:], prefix_length=30)
                last_messages = r["messages"][3:][-4:]
                if len(last_messages) == 4:
                    has_wrong_language = bot_has_wrong_language(last_messages)
                    if has_wrong_language and is_thumbs_up:
                        continue
                if is_repeating:
                    negative_records.append(r)
                    continue
                if is_thumbs_up:
                    pos_only_with_actions = config.get("pos_only_with_actions", False)
                    if pos_only_with_actions and not has_actions(last_message):
                        continue
                    pos_min_last_message_length = config.get("pos_min_last_message_length", None)
                    if pos_min_last_message_length and len(last_message) < pos_min_last_message_length:
                        continue
                    positive_records.append(r)
                    continue

                neg_max_last_message_length = config.get("neg_max_last_message_length", None)
                if neg_max_last_message_length and len(last_message) > neg_max_last_message_length:
                    continue
                negative_records.append(r)

            if not positive_records or not negative_records:
                continue
            random.shuffle(positive_records)
            random.shuffle(negative_records)
            char_name = positive_records[0]["char_name"]

            aug_negative_records = []
            for pos_record in positive_records:
                messages = pos_record["messages"]
                if len(messages) >= 7 and random.random() < config.get("aug_swap_prob", 0.0):
                    neg_messages = copy.deepcopy(messages)
                    bot_message_indices = [i for i, m in enumerate(messages) if m["role"] == "bot"][:-1]
                    swap_index = random.choice(bot_message_indices)
                    neg_messages[-1], neg_messages[swap_index] = messages[swap_index], messages[-1]
                    aug_negative_records.append({
                        "orig_messages": messages[:],
                        "messages": neg_messages,
                        "char_name": char_name,
                        "aug": "swap"
                    })
                if random.random() < config.get("aug_empty_prob", 0.0):
                    neg_messages = copy.deepcopy(messages)
                    neg_messages[-1]["content"] = ""
                    aug_negative_records.append({
                        "orig_messages": messages[:],
                        "messages": neg_messages,
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
                    aug_negative_records.append({
                        "orig_messages": messages[:],
                        "messages": neg_messages,
                        "char_name": char_name,
                        "aug": "shuffle"
                    })
                if len(messages) >= 7 and random.random() < config.get("aug_repeat_prob", 0.0):
                    neg_messages = copy.deepcopy(messages)
                    prev_bot_message = [m for i, m in enumerate(messages) if m["role"] == "bot"][-2]
                    assert messages[-1]["role"] == "bot"
                    neg_messages[-1]["content"] = prev_bot_message["content"]
                    aug_negative_records.append({
                        "orig_messages": messages[:],
                        "messages": neg_messages,
                        "char_name": char_name,
                        "aug": "repeat"
                    })
                if random.random() < config.get("aug_rm_actions", 0.0):
                    neg_messages = copy.deepcopy(messages)
                    last_message = neg_messages[-1]["content"]
                    if has_actions(last_message):
                        fixed_message = remove_actions(last_message)
                        if fixed_message:
                            neg_messages[-1]["content"] = fixed_message
                            aug_negative_records.append({
                                "orig_messages": messages[:],
                                "messages": neg_messages,
                                "char_name": char_name,
                                "aug": "rm_actions"
                            })

            make_pairwise = config.get("make_pairwise", False)

            if make_pairwise:
                for pos_record, neg_record in zip(positive_records, negative_records):
                    final_records.append({
                        "chosen_messages": pos_record["messages"],
                        "rejected_messages": neg_record["messages"],
                        "char_name": char_name
                    })
                for record in aug_negative_records:
                    final_records.append({
                        "chosen_messages": record["orig_messages"],
                        "rejected_messages": record["messages"],
                        "char_name": char_name,
                        "aug": record["aug"]
                    })
            else:
                for pos_record in positive_records:
                    final_records.append({
                        "messages": pos_record["messages"],
                        "target": 1,
                        "char_name": char_name,
                    })
                for neg_record in negative_records + aug_negative_records:
                    final_records.append({
                        "messages": neg_record["messages"],
                        "target": 0,
                        "char_name": char_name,
                    })

    random.shuffle(final_records)
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
