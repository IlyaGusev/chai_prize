from collections import Counter

from chai_prize.util.langdetect import detect_language


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
    bot_messages = [m["content"][:20] for m in messages if m["role"] == "bot"]
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


def is_single_character(char_name):
    characters = (
        "straykids groupchat",
        "Gojo,Sukuna, & Toji (CEO's)",
        "All girls sleepover",
        "two husbands",
        "a group of bullies",
        "Ghost, Soap and König",
        "könig and ghost",
        "All Female Prison",
        "Mitsuri and Shinobu",
        "Akaza, Douma, Kokushibo",
        "Boys sleepover (-W_M-)",
        "Fantasy RPG",
        "All boys school"
    )
    if " and " in char_name.lower():
        return False
    if "rpg" in char_name.lower():
        return False
    if "group" in char_name.lower():
        return False
    return char_name not in characters


def has_bad_ss(chat):
    ss = (
        " AI",
        "language model",
        "Chai"
    )
    for message in chat:
        content = message["content"]
        for s in ss:
            if s in content:
                return True
    return False


def is_not_english(chat):
    bot_messages = [m["content"] for m in chat if m["role"] == "bot"][1:]
    user_messages = [m["content"] for m in chat if m["role"] == "user"]
    if not bot_messages or not user_messages:
        return False
    bot_languages = [detect_language(message) for message in bot_messages]
    user_languages = [detect_language(message) for message in user_messages]
    bot_language = Counter(bot_languages).most_common(1)[0][0]
    user_language = Counter(user_languages).most_common(1)[0][0]
    return bot_language == user_language and user_language != "en"
