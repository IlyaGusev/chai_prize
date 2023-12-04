import sys

from chai_prize.util.io import write_jsonl, read_jsonl

input_path = sys.argv[1]
output_path = sys.argv[2]
records = read_jsonl(input_path)
chats = []
current_conv_id = None
current_chat = []
current_char_name = None
for record in records:
    if current_conv_id is not None and record["conv_id"] != current_conv_id:
        chats.append({"messages": current_chat, "conv_id": record["conv_id"], "char_name": current_char_name})
        current_chat = []
    current_char_name = record["char_name"]
    current_conv_id = record["conv_id"]
    current_chat.append({"content": record["translation"], "role": record["role"], "orig_content": record["text"]})
if current_chat:
    chats.append({"messages": current_chat, "conv_id": record["conv_id"], "char_name": current_char_name})

filtered_records = []
for record in chats:
    max_word_length = 0
    for m in record["messages"]:
        for w in m["content"].split():
            max_word_length = max(max_word_length, len(w))
    if max_word_length > 30:
        continue
    is_bad_conv = False
    for m in record["messages"]:
        content = m["content"]
        orig_content = m["orig_content"]
        if "* *" in content and "* *" not in orig_content:
            is_bad_conv = True
        if "{{" in content:
            is_bad_conv = True
        if "**" in content and "**" not in orig_content:
            is_bad_conv = True
        if "***" in content:
            is_bad_conv = True
    if is_bad_conv:
        continue
    record["messages"][1]["content"] = record["messages"][1]["content"].replace("Пользователь:", "User:")
    filtered_records.append(record)


print(chats[0])
write_jsonl(filtered_records, output_path)
