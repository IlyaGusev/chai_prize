import re
from collections import Counter

from chai_prize.util.langdetect import detect_language


UNICODE_PUNCTUATION = {
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

WHITESPACES = {" ", " ", " ", " ", " ", "　", " ", " ", " ", " "}

BAD_SS = (
    "... ...",
    ".....",
    "???????",
    "!!!!!!!",
    " AI",
    "language model",
    "Chai"
)

AI_SS = (
    "как ии",
    "как ai",
    "как аи",
    "как модель ии",
    "как алгоритм",
    "языковая модель ии",
    "как искусственный интеллект",
    "как нейросеть",
    "виртуальный ассистент",
    "виртуальный помощник",
    "как нейронная сеть",
    "онлайн-ассистент",
    "интеллектуальный помощник",
    "голосовой помощник",
    "искусственный разум",
    "компьютерная программа",
    "программный помощник",
    "представитель ии",
    "ассистент ии",
    "ии-ассистент",
    "умный искусственный интеллект",
    "помощник ai",
    "как ассистент",
    "как помощник",
    "как иси-ассистент"
    "ai помощник",
    "я являюсь искусственным интеллектом",
    "я искусственный интеллект",
    "я - искусственный интеллект",
    "я – искусственный интеллект",
    "я — искусственный интеллект",
    "я - искуственный интеллект",
    "в качестве ии",
    "в качестве искуственного интеллекта",
    "от лица ии",
    "от лица искуственного интеллекта",
    "openai",
    "chatgpt",
    "as a language model",
    "as an ai",
    " ai",
    "language model",
    "я - алгоритм",
    "я – алгоритм",
    "я - компьютерная программа",
    "я – компьютерная программа",
    "я компьютерная программа",
    "я являюсь компьютерной программой",
    "я - ai",
    "я – ai",
    "я ai",
    "я являюсь ai",
    "я - ии",
    "я – ии",
    "я ии",
    "я являюсь ии",
    "я - виртуальный помощник",
    "я – виртуальный помощник",
    "я виртуальный помощник",
    "я являюсь виртуальным помощником",
    "я - виртуальный ассистент",
    "я – виртуальный ассистент",
    "я виртуальный ассистент",
    "я являюсь виртуальным ассистентом",
    "я - программа",
    "я – программа",
    "я программа",
    "я являюсь программой",
    "я - ассистент",
    "я – ассистент",
    "я ассистент",
    "я - это искусственный интеллект",
    "я – это искусственный интеллект",
    "я – это искуственный интеллект",
    "я - это искуственный интеллект",
    "всего лишь искусственный интеллект",
    "всего лишь искуственный интеллект"
)


def normalize_whitespaces(text):
    chars = [char if char not in WHITESPACES else " " for char in text]
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
    for old, new in UNICODE_PUNCTUATION.items():
        text = text.replace(old, new)
    text = normalize_whitespaces(text)
    text = remove_quotes(text)
    text = text.replace(r"\*", "*")
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


def has_user_message(messages):
    roles = {m["role"] for m in messages}
    return "user" in roles


def generate_ngrams(elements, n: int):
    return {tuple(elements[i:i+n]) for i in range(len(elements) - n + 1)}


def has_repetition(messages, prefix_length: int = 20, max_common_ngrams: int = 6):
    bot_messages = [m["content"].lower() for m in messages if m["role"] == "bot"]

    current_ngrams = set()
    for message in bot_messages:
        new_ngrams = generate_ngrams(message.split(), n=max_common_ngrams)
        if current_ngrams.intersection(new_ngrams):
            return True
        current_ngrams.update(new_ngrams)

    start_bot_messages = [m[:prefix_length] for m in bot_messages]
    uniq_bot_messages = set(start_bot_messages)
    return len(uniq_bot_messages) < len(start_bot_messages)


def undup(records):
    new_records = {}
    for r in records:
        user_messages = [m for m in r["messages"] if m["role"] == "user"]
        bot_messages = [m for m in r["messages"] if m["role"] == "bot"]
        if user_messages:
            first_message = user_messages[0]["content"][:30]
        else:
            first_message = bot_messages[0]["content"][:30]
        new_records[(r["char_name"], first_message)] = r
    records = list(new_records.values())
    return records


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
    bot_messages = [m["content"] for m in chat if m["role"] == "bot"]
    for content in bot_messages:
        if any(s in content for s in BAD_SS):
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


def is_bot_broken(messages):
    bot_messages = [m["content"] for m in messages if m["role"] == "bot"]
    for m in bot_messages:
        words = m.split()
        for w in words:
            if len(w) >= 35:
                return True
        if not words:
            continue
        max_cnt = Counter(words).most_common()[0][1]
        if max_cnt >= 10:
            return True
    return False


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

    if is_bot_broken(chat):
        return True

    return False


def merge_bot_messages(messages):
    new_messages = []
    prev_role = None
    merge_count = 0
    for m in messages:
        role = m["role"]
        if role != "bot" or prev_role != "bot":
            new_messages.append(m)
            merge_count = 0
        else:
            assert new_messages[-1]["role"] == "bot"
            assert role == "bot"
            new_messages[-1]["content"] += "\n" + m["content"]
            merge_count += 1
            if merge_count >= 4:
                return None
        prev_role = role
    return new_messages


def has_ai_ss(messages):
    for m in messages:
        text = m["content"].lower()
        if any(ss in text for ss in AI_SS):
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


def remove_actions(message, min_length: int = 7):
    orig_message = message
    star_matches = ACTION_STAR_RE.findall(message)
    if star_matches:
        for m in star_matches:
            message = message.replace(m, " ")
    quote_matches = ACTION_DOUBLE_QUOTES_RE.findall(message)
    if quote_matches:
        for m in quote_matches:
            message = message.replace(m, " ")
    if message == orig_message:
        return None
    message = " ".join(message.split())
    return message
