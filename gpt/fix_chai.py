import fire
from itertools import chain
from tqdm import tqdm
from datasets import load_dataset

from chai_prize.util.io import read_jsonl, write_jsonl
from chai_prize.create_set import revert_flattening, clean_bot_message

KEYS = (
    "loquacity",
    "assertiveness",
    "shyness",
    "empathy",
    "kindness",
    "cruelty",
    "arrogance",
    "stubbornness",
    "humor",
    "capriciousness",
    "fragility",
    "wisdom",
    "fidelity",
    "bluntness",
    "creativity",
    "confidence",
    "integrity",
    "bellicosity",
    "patience",
    "action_level",
    "nsfw",
    "profanity",
    "user_engagement"
)

def get_chai_key(record):
    return (record["original_fields"]["conversation_id"], )


def fix_chai_output(input_path, output_path):
    rated_records = read_jsonl(input_path)
    rated_records = [r for r in rated_records if "parsed_output" in r]
    rated_records = {get_chai_key(r): r for r in rated_records}
    print(len(rated_records))

    fixed_records = []

    for key, record in rated_records.items():
        new_record = record["original_fields"]
        scores = record.pop("parsed_output")
        record.pop("output")

        for key, value in chain(scores["traits"].items(), scores["parameters"].items()):
            if key not in KEYS:
                continue
            if not key:
                continue
            if not isinstance(value, dict):
                continue
            if "score" not in value:
                continue
            if "explanation" not in value:
                continue
            new_record[key + "_score"] = value["score"]
            new_record[key + "_explanation"] = value["explanation"]
        new_record["topic"] = scores.get("topic")
        new_record["mbti_type"] = scores.get("mbti_type")
        new_record["memory"] = record["messages"][0]["content"]
        new_record["prompt"] = record["messages"][1]["content"]
        new_record["char_name"] = record["char_name"]
        for key in KEYS:
            if (key + "_score") not in new_record:
                new_record[key + "_score"] = None
            if (key + "_explanation") not in new_record:
                new_record[key + "_explanation"] = None
        fixed_records.append(new_record)
    print(len(fixed_records))
    write_jsonl(fixed_records, output_path)


if __name__ == "__main__":
    fire.Fire(fix_chai_output)
