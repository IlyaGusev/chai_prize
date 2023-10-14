def clean_bot_message(text):
    text = " ".join(text.split())
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    return text


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])


def shrink(messages, max_length):
    length = calc_max_length([{"messages": messages}])
    while length > max_length:
        messages = messages[:-2]
        length = calc_max_length([{"messages": messages}])
    return messages


def has_bot_message(messages):
    roles = {m["role"] for m in messages}
    return "bot" in roles


def has_repetition(messages):
    bot_messages = [m["content"] for m in messages if m["role"] == "bot"]
    uniq_bot_messages = set(bot_messages)
    return len(uniq_bot_messages) < len(bot_messages)


def has_correct_roles(messages):
    current_role = None
    for message in messages:
        if message.get("is_deleted", False):
            continue
        if message["role"] == current_role:
            return False
        if message["role"] not in ("system", "prompt", "bot", "user"):
            return False
        current_role = message["role"]
    return True


def has_empty_messages(messages):
    return sum([len(m["content"]) == 0 for m in messages if m["role"] == "bot"]) >= 1
