import json
from typing import List

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}: {content}</s>"


class Conversation:
    def __init__(
        self,
        prompt_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        system_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        user_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        bot_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        suffix: str = "<s>bot",
        char_name: str = None
    ):
        self.system_message_template = system_message_template
        self.user_message_template = user_message_template
        self.bot_message_template = bot_message_template
        self.prompt_message_template = prompt_message_template
        self.suffix = suffix

        self.char_name = char_name

        self.messages = []

    def get_meta(self):
        meta = dict()
        if self.char_name is not None:
            meta["char_name"] = self.char_name
        return meta

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def add_system_message(self, message):
        self.messages.append({
            "role": "system",
            "content": message
        })

    def add_prompt_message(self, message):
        self.messages.append({
            "role": "prompt",
            "content": message
        })

    def format_message(self, message):
        content = message["content"]
        if message["role"] == "system":
            return self.system_message_template.format(content=content, **self.get_meta())
        elif message["role"] == "user":
            return self.user_message_template.format(content=content, **self.get_meta())
        elif message["role"] == "prompt":
            return self.prompt_message_template.format(content=content, **self.get_meta())
        elif message["role"] == "bot":
            return self.bot_message_template.format(content=content, **self.get_meta())
        assert False

    def get_prompt(self, tokenizer, add_suffix: bool = True):
        messages = self.messages

        final_text = ""
        for message in messages:
            final_text += self.format_message(message)

        if add_suffix:
            suffix = self.suffix.format(**self.get_meta())
            final_text += suffix

        return final_text.strip()

    def iter_messages(self):
        for message in self.messages:
            yield self.format_message(message), message["role"]

    @classmethod
    def from_template(cls, file_name, **kwargs):
        with open(file_name, encoding="utf-8") as r:
            template = json.load(r)
        return Conversation(
            **template,
            **kwargs
        )

    def expand(self, messages):
        if messages[0]["role"] == "system":
            self.messages = []

        for message in messages:
            if message["role"] == "user":
                self.add_user_message(message["content"])
            elif message["role"] == "bot":
                self.add_bot_message(message["content"])
            elif message["role"] == "system":
                self.add_system_message(message["content"])
            elif message["role"] == "prompt":
                self.add_prompt_message(message["content"])
            else:
                assert False
