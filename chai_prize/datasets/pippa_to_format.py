import sys

from chai_prize.util.io import write_jsonl, read_jsonl

input_path = sys.argv[1]
output_path = sys.argv[2]


new_records = []
records = read_jsonl(input_path)
for record in records:
    messages = record["messages"]
    if "parsed_output" not in record:
        continue
    new_record = {
        "gpt_35_turbo_result": record["parsed_output"]["result"],
        "gpt_35_turbo_explanation": record["parsed_output"]["explanation"],
        "translation_model": "nllb-200-3.3B",
        "bot_name": record["char_name"],
        "bot_definitions": messages[1]["content"],
        "orig_bot_definitions": messages[1]["orig_content"],
        "bot_description": messages[0]["content"],
        "orig_bot_description": messages[0]["orig_content"],
        "conversation": [{"message": m["content"], "is_human": (m["role"] == "user")} for m in messages[2:]],
        "orig_conversation": [{"message": m["orig_content"], "is_human": (m["role"] == "user")} for m in messages[2:]],
    }
    new_records.append(new_record)
write_jsonl(new_records, output_path)
