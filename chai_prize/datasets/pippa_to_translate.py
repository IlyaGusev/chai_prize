import sys

from datasets import load_dataset
from tqdm import tqdm

from chai_prize.util.io import write_jsonl
from chai_prize.create_set import revert_flattening, clean_bot_message
from chai_prize.util.data import is_bad_chat, bot_has_wrong_language, is_not_english


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


output_path = sys.argv[1]

records = []
for row in tqdm(load_dataset("PygmalionAI/PIPPA", split="train")):
    context = row["bot_description"]
    char_name = row["bot_name"].strip()
    messages = revert_flattening(row["conversation"])

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

    if is_bad_chat(chat):
        continue

    if bot_has_wrong_language(chat):
        continue

    if is_not_english(chat):
        continue

    row["messages"] = chat
    conv_key = hash(get_pippa_key(row))
    for i, message in enumerate(chat):
        records.append({
            "text": message["content"],
            "role": message["role"],
            "conv_id": conv_key,
            "message_num": i,
            "text_id": hash((message["content"], i)),
            "char_name": char_name
        })
write_jsonl(records, output_path)
