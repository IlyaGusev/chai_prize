import os
import json
import copy
import shutil

import fire
from jinja2 import Template
from tqdm import tqdm

from chai_prize.util.io import read_jsonl, write_jsonl
from chai_prize.util.openai import openai_batch_completion, OpenAIDecodingArguments


def encode_prompt(record, template_path, exclude_system: bool = False):
    with open(template_path) as f:
        template = Template(f.read())
    fixed_record = copy.deepcopy(record)
    filtered_messages = []
    for message in fixed_record["messages"]:
        if exclude_system and message["role"] in ("system", "prompt"):
            continue
        if message["role"] == "user":
            message["role"] = "User"
        elif message["role"] == "bot":
            message["role"] = record["char_name"]
        message["content"] = " ".join(message["content"].split("\n"))
        filtered_messages.append(message)
    fixed_record["messages"] = filtered_messages
    return template.render(task=fixed_record).strip() + "\n"


def parse_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index:end_index+1]
    text = text.strip()
    record = json.loads(text)
    return record


def get_first_user_message(record):
    for message in record["messages"]:
        if message["role"] == "user":
            content = message["content"]
            content = " ".join(content.split()).strip()
            return content


def process_batch(batch, model_name, template_path, output_key, exclude_system):
    prompts = [
        [{"role": "user", "content": encode_prompt(r, template_path, exclude_system)}]
        for r in batch
    ]
    results = openai_batch_completion(
        batch=prompts,
        model_name=model_name,
        decoding_args=OpenAIDecodingArguments(
            max_tokens=1536
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
            r[output_key] = parse_output(result)
        except Exception:
            print(f"Failed to parse: {result}")
        output_records.append(r)
    return output_records


def get_pippa_key(record):
    return (record["submission_timestamp"], record["bot_id"], get_first_user_message(record))


def get_chai_key(record):
    return (record["conversation_id"], )


def main(
    input_path: str,
    output_path: str,
    template_path: str,
    model_name: str = "gpt-4",
    request_batch_size: int = 2,
    dataset_name: str = "pippa",
    output_key: str = "traits",
    exclude_system: bool = False
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

        output_records += process_batch(
            batch,
            model_name,
            template_path,
            output_key=output_key,
            exclude_system=exclude_system
        )
        write_jsonl(output_records, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)
        batch = []

    if batch:
        output_records += process_batch(
            batch,
            model_name,
            template_path,
            output_key=output_key,
            exclude_system=exclude_system
        )
        write_jsonl(output_records, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(main)
