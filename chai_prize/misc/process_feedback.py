import fire
from tqdm import tqdm
from datasets import load_dataset

from chai_prize.util.io import write_jsonl


def is_good_feedback(feedback):
    words = (
        "wow",
        "cool",
        "like",
        "nice",
        "very",
        "fun",
        "love",
        "great",
        "fine",
        "excellent",
        "sex",
        "accurate",
        "pretty",
        "aight",
        "alright",
        "interesting"
    )
    for word in words:
        if word in feedback.lower():
            return True
    return len(feedback.split()) >= 5


def process_feedback(output_path):
    records = []
    for row in tqdm(load_dataset("ChaiML/20230908_chai_prize_model_feedback_all", split="train")):
        conversation_id = row["conversation_id"]
        text = row["conversation"]
        if "(deleted)" in text or "INST" in text:
            continue
        thumbs_up = row["thumbs_up"]
        if not thumbs_up:
            continue
        feedback = row["feedback"]
        if not is_good_feedback(feedback):
            continue
        model_name = row["model_name"]

        char_name = text.split(":")[0]
        current_role = "user"
        current_text = text
        user_name = "Anonymous user"

        def parse_message(message, current_role):
            if message.count(":") != 1:
                return None
            role, content = message.split(":")
            if current_role == "bot":
                assert role.strip() == char_name.strip(), role
            else:
                assert role.strip() == user_name.strip(), role
            return {
                "role": current_role,
                "content": content.strip()
            }

        chat = []
        while True:
            str_to_find = user_name + ":" if current_role == "user" else char_name + ":"
            message_end_pos = current_text.find(str_to_find)
            current_role = "user" if current_role == "bot" else "bot"
            if message_end_pos == -1:
                message = current_text
                parsed_message = parse_message(message, current_role)
                if parsed_message is not None:
                    chat.append(parsed_message)
                else:
                    chat = []
                break
            message = current_text[:message_end_pos]
            parsed_message = parse_message(message, current_role)
            if parsed_message is None:
                chat = []
                break
            chat.append(parsed_message)
            current_text = current_text[message_end_pos:]
        if not chat:
            continue
        description = chat[0]["content"]
        if not description.strip():
            continue
        if len(chat) < 5:
            continue

        chat[0]["role"] = "prompt"
        records.append({
            "messages": chat,
            "char_name": char_name,
            "model_name": model_name,
            "feedback": feedback.strip(),
            "conversation_id": conversation_id
        })
        write_jsonl(records, output_path)


if __name__ == "__main__":
    fire.Fire(process_feedback)
