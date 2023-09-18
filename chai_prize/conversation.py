import json
from typing import List

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"

class Conversation:
    def __init__(
        self,
        prompt_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        system_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        user_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        bot_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        system_prompt: str = None,
        system_role: str = "system",
        user_role: str = "user",
        bot_role: str = "bot",
        prompt_role: str = "prompt",
        suffix: str = "<s>bot"
    ):
        self.system_message_template = system_message_template
        self.user_message_template = user_message_template
        self.bot_message_template = bot_message_template
        self.prompt_message_template = prompt_message_template
        self.system_role = system_role
        self.user_role = user_role
        self.bot_role = bot_role
        self.prompt_role = prompt_role
        self.suffix = suffix
        self.messages = []
        if system_prompt:
            self.messages = [{
                "role": self.system_role,
                "content": system_prompt
            }]

    def add_user_message(self, message):
        self.messages.append({
            "role": self.user_role,
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": self.bot_role,
            "content": message
        })

    def add_prompt_message(self, message):
        self.messages.append({
            "role": self.prompt_role,
            "content": message
        })

    def format_message(self, message):
        if message["role"] == self.system_role:
            return self.system_message_template.format(**message)
        if message["role"] == self.user_role:
            return self.user_message_template.format(**message)
        if message["role"] == self.prompt_role:
            return self.prompt_message_template.format(**message)
        return self.bot_message_template.format(**message)

    def get_prompt(self, tokenizer,add_suffix: bool = True):
        messages = self.messages

        final_text = ""
        for message in messages:
            final_text += self.format_message(message)

        if add_suffix:
            final_text += self.suffix

        return final_text.strip()

    def iter_messages(self):
        for message in self.messages:
            yield self.format_message(message), message["role"]

    @classmethod
    def from_template(cls, file_name):
        with open(file_name, encoding="utf-8") as r:
            template = json.load(r)
        return Conversation(
            **template
        )

    def expand(self, messages, role_mapping = None):
        if not role_mapping:
            role_mapping = dict()

        if messages[0]["role"] == "system":
            self.messages = []

        for message in messages:
            self.messages.append({
                "role": role_mapping.get(message["role"], message["role"]),
                "content": message["content"]
            })
