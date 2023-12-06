def parse_chai_conversation(text):
    text = text.strip()

    if "'s Persona" in text[:100]:
        parts = text.split("'s Persona")
        char_name = parts[0].strip()
        text = parts[1].strip()

        parts = text.split("####")
        system_message = parts[0].strip()
        text = parts[1].strip()

        if "<START>" in text:
            parts = text.split("<START>")
            prompt_message = parts[0].strip()
            text = parts[1].strip()
    else:
        char_name = text.split(":")[0].strip()

    lines = []
    role = "bot"
    is_deleted = False

    deleted_start = f"{char_name} (deleted):"
    char_start = f"{char_name}:"
    user_start1 = "Anonymous user:"
    user_start2 = "You:"

    for line in text.split("\n"):
        line = line.strip()

        current_start = None
        for start in (deleted_start, char_start, user_start1, user_start2):
            if line.startswith(start):
                current_start = start

        if current_start is None:
            lines.append(line)
            continue

        if lines:
            yield {"role": role, "content": "\n".join(lines).strip(), "is_deleted": is_deleted}

        lines = [line.replace(current_start, "").strip()]
        role = "bot" if current_start not in (user_start1, user_start2) else "user"
        is_deleted = current_start == deleted_start

    if lines:
        yield {"role": role, "content": "\n".join(lines).strip(), "is_deleted": is_deleted}


def is_whitelisted_model(model_name):
    models = (
        "ilyagusev",
        "khoantap",
        "anhnv125",
        "tokenbender",
        "tehvenom",
        "khanhnto",
        "liquac09",
        "monkeydddd",
        "alkahestry",
        "the-face-of-goonery",
        "jnewber",
        "chargoddard",
        "ansoi"
    )
    for model in models:
        if model in model_name:
            return True
    return False


def is_good_feedback(feedback):
    bad_words = (
        "meh",
        "idk",
        "mid",
        "repeat",
        "mediocre",
        "average"
    )
    for word in bad_words:
        if word in feedback.lower():
            return False
    if len(feedback.split()) >= 3:
        return True
    words = (
        "cute",
        "good",
        "gud",
        "yes",
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
        "interesting",
        "better"
        "goof",
        "comforing",
        "comforting",
        "ok",
        "okay",
        "amazing",
        "damn",
        "gusta",
        "awesome",
        "not bad",
        "hot",
        "lol",
        "xd",
        "buena",
        "perfect",
        "норм",
        "классно",
        "epic",
        "uwu",
        "🌟",
        "best",
        "bien",
        "ดี",
        ":)",
        "⭐",
        "👍🏻",
        "👍",
        "❤️ ",
        "decent",
        "beautiful",
        "god",
        "funny",
        "smash",
        "kind",
        "bueno",
        "normal",
        "fantastic",
        "something",
        "handsome",
        "reasonable",
        "agradable",
        "genial",
        "better",
        "delightful",
        "intresting",
        "super",
        "omg"
    )
    for word in words:
        if word in feedback.lower():
            return True
    return False


TEST_TEXT = """
Bodyguard Konig: *your father wants you under watch after a pattern of
reckless behaviours and partying.
just for your luck, he hired a 6’10 Austrian man.
*Konig looks at you, shaking his head slightly.* “I do not want you going out.”
*he says, his arms crossing over his chest* “stay home… Bitte…”
Anonymous user: huh? why
Bodyguard Konig (deleted): *Konig looks at you, shaking his head slightly.*
Bodyguard Konig: “I do not want you going out.” *he says, his arms crossing over his chest*
Anonymous user: why!
Bodyguard Konig: “I do not want you going out.”
"""

MESSAGE_0 = """
*your father wants you under watch after a pattern of
reckless behaviours and partying.
just for your luck, he hired a 6’10 Austrian man.
*Konig looks at you, shaking his head slightly.* “I do not want you going out.”
*he says, his arms crossing over his chest* “stay home… Bitte…”
""".strip()

if __name__ == "__main__":
    messages = list(parse_chai_conversation(TEST_TEXT))
    assert messages[0]["role"] == "bot"
    assert messages[0]["content"] == MESSAGE_0
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "huh? why"
    assert messages[2]["role"] == "bot"
    assert messages[2]["content"] == "*Konig looks at you, shaking his head slightly.*"
    assert messages[2]["is_deleted"]
    assert messages[3]["role"] == "bot"
    assert messages[3]["content"] == "“I do not want you going out.” *he says, his arms crossing over his chest*"
    assert messages[4]["role"] == "user"
    assert messages[4]["content"] == "why!"
    assert messages[5]["role"] == "bot"
    assert messages[5]["content"] == "“I do not want you going out.”"
