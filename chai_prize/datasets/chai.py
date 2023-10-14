def parse_chai_conversation(text):
    text = text.strip()
    char_name = text.split(":")[0].strip()
    user_name = "Anonymous user"

    messages = []

    lines = []
    role = "bot"
    is_deleted = False

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(char_name + " (deleted):"):
            if lines:
                yield {"role": role, "content": "\n".join(lines).strip(), "is_deleted": is_deleted}
            role, lines = "bot", [line.replace(char_name + " (deleted):", "").strip()]
            is_deleted = True
        elif line.startswith(char_name + ":"):
            if lines:
                yield {"role": role, "content": "\n".join(lines).strip(), "is_deleted": is_deleted}
            role, lines = "bot", [line.replace(char_name + ":", "").strip()]
            is_deleted = False
        elif line.startswith(user_name + ":"):
            if lines:
                yield {"role": role, "content": "\n".join(lines).strip(), "is_deleted": is_deleted}
            role, lines = "user", [line.replace(user_name + ":", "").strip()]
            is_deleted = False
        else:
            lines.append(line)
    if lines:
        yield {"role": role, "content": "\n".join(lines).strip(), "is_deleted": is_deleted}


TEST_TEXT = """
Bodyguard Konig: *your father wants you under watch after a pattern of reckless behaviours and partying.
just for your luck, he hired a 6’10 Austrian man.
*Konig looks at you, shaking his head slightly.* “I do not want you going out.”
*he says, his arms crossing over his chest* “stay home… Bitte…”
Anonymous user: huh? why
Bodyguard Konig (deleted): *Konig looks at you, shaking his head slightly.* “I do not want you going out.”
Bodyguard Konig: *Konig looks at you, shaking his head slightly.* “I do not want you going out.” *he says, his arms crossing over his chest*
Anonymous user: why!
Bodyguard Konig: *Konig looks at you, shaking his head slightly.* “I do not want you going out.”
"""

if __name__ == "__main__":
    messages = list(parse_chai_conversation(TEST_TEXT))
    assert messages[0]["role"] == "bot"
    assert messages[0]["content"] == """*your father wants you under watch after a pattern of reckless behaviours and partying.
just for your luck, he hired a 6’10 Austrian man.
*Konig looks at you, shaking his head slightly.* “I do not want you going out.”
*he says, his arms crossing over his chest* “stay home… Bitte…”"""
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "huh? why"
    assert messages[2]["role"] == "bot"
    assert messages[2]["content"] == "*Konig looks at you, shaking his head slightly.* “I do not want you going out.”"
    assert messages[2]["is_deleted"] == True
    assert messages[3]["role"] == "bot"
    assert messages[3]["content"] == "*Konig looks at you, shaking his head slightly.* “I do not want you going out.” *he says, his arms crossing over his chest*"
    assert messages[4]["role"] == "user"
    assert messages[4]["content"] == "why!"
    assert messages[5]["role"] == "bot"
    assert messages[5]["content"] == "*Konig looks at you, shaking his head slightly.* “I do not want you going out.”"
