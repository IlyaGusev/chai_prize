from typing import Optional, List

from chaiverse.formatters import PromptFormatter


class CustomFormatterV3(PromptFormatter):
    memory_template: str = "{bot_name}'s Persona: {memory}\n####\n"
    prompt_template: str = "{prompt}\n<START>\n"
    bot_template: str = "{bot_name}: {message}\n"
    user_template: str = "User: {message}\n"
    response_template: str = "{bot_name}:"
