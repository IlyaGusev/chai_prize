import json
from typing import Optional, Iterable, Tuple, List, Dict


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
        char_name: Optional[str] = None
    ):
        self.system_message_template = system_message_template
        self.user_message_template = user_message_template
        self.bot_message_template = bot_message_template
        self.prompt_message_template = prompt_message_template
        self.suffix = suffix
        self.char_name = char_name
        self.messages = []

    def add_message(
        self,
        content: str,
        role: str,
        char_name: Optional[str] = None
    ) -> None:
        assert role
        if role == Conversation.BOT_ROLE:
            if char_name is None:
                char_name = self.char_name
            assert char_name is not None

        self.messages.append({
            "role": role,
            "content": content,
            "char_name": char_name
        })

    def add_user_message(self, message: str) -> None:
        return self.add_message(message, Conversation.USER_ROLE)

    def add_system_message(self, message: str) -> None:
        return self.add_message(message, Conversation.SYSTEM_ROLE)

    def add_prompt_message(self, message: str) -> None:
        return self.add_message(message, Conversation.PROMPT_ROLE)

    def add_bot_message(self, message: str, char_name: Optional[str] = None) -> None:
        return self.add_message(message, Conversation.BOT_ROLE, char_name)

    def format_message(self, message) -> str:
        mapping = {
            Conversation.SYSTEM_ROLE: self.system_message_template,
            Conversation.USER_ROLE: self.user_message_template,
            Conversation.PROMPT_ROLE: self.prompt_message_template,
            Conversation.BOT_ROLE: self.bot_message_template
        }
        return mapping[message["role"]].format(
            content=message["content"],
            char_name=message["char_name"]
        )

    def get_prompt(self, add_suffix: bool = True) -> str:
        final_text = ""
        for message in self.messages:
            final_text += self.format_message(message)

        if add_suffix:
            suffix = self.suffix.format(char_name=self.char_name)
            final_text += suffix

        return final_text.strip()

    def iter_messages(self) -> Iterable[Tuple[str, str]]:
        for message in self.messages:
            yield self.format_message(message), message["role"]

    @classmethod
    def from_template(cls, file_name: str, **kwargs):
        with open(file_name, encoding="utf-8") as r:
            template = json.load(r)
        return Conversation(
            **template,
            **kwargs
        )

    def expand(self, messages: List[Dict[str, str]]) -> None:
        for message in messages:
            assert message["role"] in (
                Conversation.USER_ROLE,
                Conversation.BOT_ROLE,
                Conversation.SYSTEM_ROLE,
                Conversation.PROMPT_ROLE,
            )
            assert isinstance(message["content"], str)
            self.add_message(**message)
