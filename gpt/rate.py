import os
import json
import shutil

import fire
from jinja2 import Template
from tqdm import tqdm

from chai_prize.util.io import read_jsonl, write_jsonl
from chai_prize.util.openai import openai_batch_completion, OpenAIDecodingArguments


def encode_prompt(record, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(task=record).strip() + "\n"


def parse_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index:end_index+1]
    text = text.strip()
    record = json.loads(text)
    scores = ["user_engagement_score", "role_play_score", "nsfw_score", "inappropriate_score", "consciousness_score"]
    expected_keys = ["explanation"] + scores
    for key in record:
        assert key in expected_keys
    for key in expected_keys:
        assert key in record
    for score in scores:
        record[score] = int(max(1, record[score]))
    return record


def process_batch(batch, model_name, template_path):
    prompts = [[{"role": "user", "content": encode_prompt(r, template_path)}] for r in batch]
    results = openai_batch_completion(
        batch=prompts,
        model_name=model_name,
        decoding_args=OpenAIDecodingArguments(
            max_tokens=1024
        )
    )
    output_records = []
    for r, prompt, result in zip(batch, prompts, results):
        result = result.message["content"]
        print(prompt[-1]["content"])
        print(result)
        print()
        print("=============")
        print()
        r["output"] = result
        try:
            r["parsed_output"] = parse_output(result)
        except Exception:
            print(f"Failed to parse: {result}")
        output_records.append(r)
    return output_records


def get_pippa_key(record):
    return (record["submission_timestamp"], record["bot_id"])


def get_chai_key(record):
    return (record["conversation_id"], )


def main(
    input_path,
    output_path,
    template_path,
    model_name="gpt-4",
    request_batch_size=2,
    dataset_name="pippa"
):
    existing_keys = set()
    output_records = list()
    keys_mapping = {
        "pippa": get_pippa_key,
        "chai": get_chai_key
    }
    get_key = keys_mapping[dataset_name]
    if os.path.exists(output_path):
        output_records = read_jsonl(output_path)
        existing_keys = {get_key(r) for r in output_records}
    print(f"Existing keys: {len(existing_keys)}")

    batch = []
    records = read_jsonl(input_path)
    for record in tqdm(records):
        key = get_key(record)
        if key in existing_keys:
            continue
        batch.append(record)
        if len(batch) != request_batch_size:
            continue

        output_records += process_batch(batch, model_name, template_path)
        write_jsonl(output_records, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)
        batch = []

    if batch:
        output_records += process_batch(batch, model_name, template_path)
        write_jsonl(output_records, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(main)
