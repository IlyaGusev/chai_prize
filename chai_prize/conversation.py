import json


class Conversation:
    USER_ROLE: str = "user"
    BOT_ROLE: str = "bot"
    SYSTEM_ROLE: str = "system"
    PROMPT_ROLE: str = "prompt"

    def __init__(
        self,
        prompt_message_template: str,
        system_message_template: str,
        user_message_template: str,
        bot_message_template: str,
        suffix: str,
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

    def add_message(self, message, role, char_name = None):
        self.messages.append({
            "role": role,
            "content": message,
            "char_name": char_name
        })

    def add_user_message(self, message):
        return self.add_message(message, Conversation.USER_ROLE)

    def add_bot_message(self, message, char_name = None):
        return self.add_message(message, Conversation.BOT_ROLE, char_name)

    def add_system_message(self, message):
        return self.add_message(message, Conversation.SYSTEM_ROLE)

    def add_prompt_message(self, message):
        return self.add_message(message, Conversation.PROMPT_ROLE)

    def format_message(self, message):
        mapping = {
            Conversation.SYSTEM_ROLE: self.system_message_template,
            Conversation.USER_ROLE: self.user_message_template,
            Conversation.PROMPT_ROLE: self.prompt_message_template,
            Conversation.BOT_ROLE: self.bot_message_template
        }
        content = message["content"]
        char_name = message["char_name"]
        if char_name is None:
            return mapping[message["role"]].format(content=content, **self.get_meta())
        return mapping[message["role"]].format(content=content, char_name=char_name)

    def get_prompt(self, add_suffix: bool = True):
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
        for message in messages:
            assert message["role"] in (
                Conversation.USER_ROLE,
                Conversation.BOT_ROLE,
                Conversation.SYSTEM_ROLE,
                Conversation.PROMPT_ROLE,
            )
            self.add_message(message["content"], message["role"], message.get("char_name"))
