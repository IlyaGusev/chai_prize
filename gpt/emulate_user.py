import os
import json
import copy
import shutil
from statistics import mean

import requests
import fire
from jinja2 import Template
from tqdm import tqdm

from chai_prize.conversation import Conversation
from chai_prize.util.io import read_jsonl, write_jsonl
from chai_prize.util.openai import openai_completion, OpenAIDecodingArguments


def generate(url, text, top_p: float = 0.9, top_k: int = 30, temp: float = 0.5, rep_penalty: float = 1.2):
    data = {
        "inputs": text,
        "parameters": {
            "max_new_tokens": 200,
            "repetition_penalty": rep_penalty,
            "do_sample": True,
            "temperature": temp,
            "seed": 42,
            "top_p": top_p,
            "top_k": top_k,
            "watermark": False,
            "stop": ["\n"]
        },
    }
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(url=url, json=data, headers=headers)
    data = response.json()
    out_text = data["generated_text"].strip()
    return out_text


def answer_as_bot(url, history, char_name, system_prompt, example_prompt, initial_message):
    conversation = Conversation(
        char_name=char_name,
        prompt_message_template="{content}\n<START>\n",
        system_message_template="{char_name}'s Persona: {content}\n####\n",
        user_message_template="User: {content}\n",
        bot_message_template="{char_name}: {content}\n",
        suffix="{char_name}:",
    )
    conversation.add_system_message(system_prompt)
    conversation.add_prompt_message(example_prompt)
    conversation.add_bot_message(initial_message)
    for message in history:
        if message["role"] == "user":
            conversation.add_user_message(message["content"])
        else:
            conversation.add_bot_message(message["content"])
    prompt = conversation.get_prompt()
    while True:
        try:
            output = generate(url, prompt)
        except Exception as e:
            continue
        break
    return output


def encode_prompt(record, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(**record).strip() + "\n"


def parse_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index:end_index+1]
    text = text.strip()
    record = json.loads(text)
    return record


def process_record(record, template_path, model_name):
    prompt = [{"role": "user", "content": encode_prompt(record, template_path)}]
    while True:
        try:
            result = openai_completion(
                prompt,
                model_name=model_name,
                decoding_args=OpenAIDecodingArguments(
                    max_tokens=1536
                ),
                sleep_time=10
            )
            print(prompt[0]["content"])
            print(result)
            print()
            print("=============")
            print()
            output = parse_output(result)
            break
        except Exception as e:
            continue
    return output


def main(
    template_path: str,
    settings_path: str,
    output_path: str,
    url: str,
    model_name: str = "gpt-4-1106-preview",
    request_batch_size: int = 1,
):
    with open(settings_path) as r:
        settings = json.load(r)
    char_records = settings["characters"]
    situations = settings["situations"]

    outputs = []
    for char_record in char_records:
        for situation in situations:
            messages = [{"role": "bot", "content": char_record["initial_message"]}]
            for _ in range(settings["num_turns"]):
                output = process_record({
                    "char_description": char_record["system_prompt"],
                    "situation": situation,
                    "messages": messages
                }, template_path=template_path, model_name=model_name)
                messages.append({"role": "user", "content": output["next_user_utterance"]})
                bot_message = answer_as_bot(
                    url,
                    messages,
                    **char_record
                )
                print(bot_message)
                print("@@@@")
                messages.append({"role": "bot", "content": bot_message})
            output = process_record({
                "char_description": char_record["system_prompt"],
                "situation": situation,
                "messages": messages
            }, template_path=template_path, model_name=model_name)
            output.pop("next_user_utterance")
            output["messages"] = messages
            outputs.append(output)

    def calc_score(o):
        return o["stay_in_character_score"] / 3 + o["grammar_score"] / 3 + o["entertainment_score"] / 3

    final_score = mean([calc_score(o) for o in outputs])
    with open(output_path, "w") as w:
        w.write(json.dumps({"outputs": outputs, "final_score": final_score}, ensure_ascii=False, indent=4))



if __name__ == "__main__":
    fire.Fire(main)
