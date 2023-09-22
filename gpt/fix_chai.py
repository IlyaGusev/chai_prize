import fire
from tqdm import tqdm
from datasets import load_dataset

from chai_prize.util.io import read_jsonl, write_jsonl
from chai_prize.create_set import revert_flattening, clean_bot_message


def get_chai_key(record):
    return (record["conversation_id"], )


def fix_chai_output(input_path, output_path):
    rated_records = read_jsonl(input_path)
    rated_records = [r for r in rated_records if "parsed_output" in r]
    rated_records = {get_chai_key(r): r for r in rated_records}
    print(len(rated_records))

    fixed_records = []
    for key, record in rated_records.items():
        scores = record.pop("parsed_output")
        for key, value in scores.items():
            record[key] = value
        if sum(["START" in m["content"] for m in record["messages"]]) > 0:
            continue

        record["messages"][1]["role"] = "bot"
        record["messages"].insert(1, {"role": "prompt", "content": ""})
        record["scores_explanation"] = record.pop("explanation")
        record.pop("output")
        record.pop("source")
        fixed_records.append(record)
    print(len(fixed_records))
    write_jsonl(fixed_records, output_path)


if __name__ == "__main__":
    fire.Fire(fix_chai_output)
