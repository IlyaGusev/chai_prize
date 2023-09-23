import fire
from tqdm import tqdm
from datasets import load_dataset

from chai_prize.util.io import read_jsonl, write_jsonl
from chai_prize.create_set import revert_flattening, clean_bot_message


def clean_message(message):
    message = " ".join(message.split())
    return message


def get_pippa_key(record):
    user_message = None
    user_message_idx = 0
    for i, message in enumerate(record["messages"]):
        if message["role"] == "user":
            user_message = clean_message(message["content"])
            user_message_idx = i
            break
    bot_message = record["messages"][user_message_idx+1]["content"]
    bot_message = clean_message(bot_message)
    return (record["submission_timestamp"], record["bot_id"], user_message, bot_message)


def merge_pippa_output(input_path, original_path, output_path):
    rated_records = read_jsonl(input_path)
    rated_records = [r for r in rated_records if "parsed_output" in r]
    rated_records = {get_pippa_key(r): r for r in rated_records}
    print(len(rated_records))

    merged_records = []
    for row in tqdm(read_jsonl(original_path)):
        bot_id = row["bot_id"]
        submission_timestamp = int(row["submission_timestamp"]) // 1000

        if len(row["conversation"]) < 3:
            continue
        bot_message = clean_message(clean_bot_message(row["conversation"][2]["message"]))
        user_message = clean_message(row["conversation"][1]["message"])
        key = (submission_timestamp, bot_id, user_message, bot_message)
        if key not in rated_records:
            continue

        scores = rated_records[key]["parsed_output"]
        for key, value in scores.items():
            row[key] = value
        row["scores_explanation"] = row.pop("explanation")
        merged_records.append(row)
    print(len(merged_records))
    write_jsonl(merged_records, output_path)


if __name__ == "__main__":
    fire.Fire(merge_pippa_output)