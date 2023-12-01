from collections import Counter

from chai_prize.util.langdetect import detect_language

unicode_punctuation = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}

whitespaces = {" ", " ", " ", " ", " ", "　", " ", " ", " ", " "}


def normalize_whitespaces(text):
    chars = [char if char not in whitespaces else " " for char in text]
    text = "".join(chars)
    return text


def remove_quotes(text):
    if len(text) <= 2:
        return text
    if text[0] == '"' and text[-1] == '"' and '"' not in text[1:-1]:
        return text[1:-1]
    return text


def clean_bot_message(text):
    text = " ".join(text.split())
    for old, new in unicode_punctuation.items():
        text = text.replace(old, new)
    text = normalize_whitespaces(text)
    text = remove_quotes(text)
    text = text.replace("\*", "*")
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


def remove_trailing_user_messages(chat):
    while chat and (chat[-1]["role"] == "user" or chat[-1].get("is_deleted", False)):
        chat.pop()
    if chat:
        assert chat[-1]["role"] != "user"
    return chat


def shrink(messages, max_length):
    length = calc_max_length([{"messages": messages}])
    while length > max_length:
        messages.pop()
        length = calc_max_length([{"messages": messages}])
    remove_trailing_user_messages(messages)
    return messages


def has_bot_message(messages):
    roles = {m["role"] for m in messages}
    return "bot" in roles


def has_repetition(messages, prefix_length: int = 20):
    bot_messages = [m["content"][:prefix_length] for m in messages if m["role"] == "bot"]
    uniq_bot_messages = set(bot_messages)
    return len(uniq_bot_messages) < len(bot_messages)


def has_correct_roles(messages):
    for message in messages:
        if message["role"] not in ("system", "prompt", "bot", "user"):
            return False
    return True


def has_empty_bot_messages(messages):
    return sum([len(m["content"].strip()) == 0 for m in messages if m["role"] == "bot"]) >= 1


def is_single_character(char_name):
    tokens = {token.lower() for token in char_name.split()}
    if " and " in char_name.lower():
        return False
    if "&" in char_name.lower():
        return False
    if "two" in tokens:
        return False
    if "all" in tokens:
        return False
    if "rpg" in char_name.lower():
        return False
    if "group" in char_name.lower():
        return False
    if "game" in tokens:
        return False
    characters = (
        "Akaza, Douma, Kokushibo",
        "Boys sleepover (-W_M-)"
    )
    return char_name not in characters


def has_bad_keywords(chat):
    ss = (
        " AI",
        "language model",
        "Chai"
    )
    for message in chat:
        content = message["content"]
        if any(s in content for s in ss):
            return True
    return False


def calc_main_language(chat, role):
    messages = [m["content"] for m in chat if m["role"] == role][1:]
    if not messages:
        return None
    languages = [detect_language(message) for message in messages]
    language = Counter(languages).most_common(1)[0][0]
    return language


def is_not_english(chat):
    bot_language = calc_main_language(chat, "bot")
    user_language = calc_main_language(chat, "user")
    return bot_language == user_language and user_language != "en"


def bot_has_wrong_language(chat):
    bot_language = calc_main_language(chat, "bot")
    user_language = calc_main_language(chat, "user")
    return bot_language != user_language


def is_bad_chat(chat):
    if not chat:
        return True

    if not has_correct_roles(chat):
        return True

    if has_repetition(chat):
        return True

    if has_empty_bot_messages(chat):
        return True

    if has_bad_keywords(chat):
        return True

    if not has_bot_message(chat):
        return True

    return False


ACTION_STAR_RE = re.compile(r'\*[^\*]+\*')
ACTION_FANCY_QUOTES_RE = re.compile(r'“[^”]+”')
ACTION_DOUBLE_QUOTES_RE = re.compile(r'"[^"]+"')


def has_actions(message, min_length: int = 7):
    star_matches = ACTION_STAR_RE.findall(message)
    if star_matches:
        length = max(len(m) for m in star_matches)
        if length >= min_length:
            return True
    fancy_matches = ACTION_FANCY_QUOTES_RE.findall(message)
    if fancy_matches:
        length = sum(len(m) for m in fancy_matches)
        action_length = len(message) - length
        if action_length >= min_length:
            return True
    quote_matches = ACTION_DOUBLE_QUOTES_RE.findall(message)
    if quote_matches:
        length = sum(len(m) for m in quote_matches)
        action_length = len(message) - length
        if action_length >= min_length:
            return True
    return False
