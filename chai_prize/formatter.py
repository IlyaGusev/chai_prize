from typing import Optional, List

from chai_guanaco.formatters import PromptFormatter


class CustomFormatterV1(PromptFormatter):
    memory_template = "<s>system\nYou are {bot_name}. {memory}</s>"
    prompt_template = "<s>prompt\n{prompt}</s>"
    bot_template = "<s>bot\n{message}</s>"
    user_template = "<s>user\n{message}</s>"
    response_template = "<s>bot\n"


class CustomFormatterV2(PromptFormatter):
    memory_template = "<s>system: You are {bot_name}. {memory}</s>"
    prompt_template = "<s>prompt: {prompt}</s>"
    bot_template = "<s>{bot_name}: {message}</s>"
    user_template = "<s>User: {message}</s>"
    response_template = "<s>{bot_name}:"


full_memory_template = "{{bot_name}}'s Persona: {prefix}{{memory}}{suffix}{controls}\n####\n"
full_prompt_template = "{prefix}{{prompt}}{suffix}\n<START>\n"

class CustomFormatterV3(PromptFormatter):
    memory_template = full_memory_template.format(controls="", prefix="", suffix="")
    prompt_template = full_prompt_template.format(prefix="", suffix="")
    bot_template = "{bot_name}: {message}\n"
    user_template = "User: {message}\n"
    response_template = "{bot_name}:"

    def __init__(
        self,
        attributes: Optional[List[str]] = None,
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        memory_prefix: str = "",
        memory_suffix: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        if prompt_prefix or prompt_suffix:
            prompt_prefix = prompt_prefix.strip()
            if prompt_prefix:
                prompt_prefix += "\n"
            prompt_suffix = prompt_suffix.strip()
            if prompt_suffix:
                prompt_suffix = "\n" + prompt_suffix
            self.prompt_template = full_prompt_template.format(
                prefix=prompt_prefix,
                suffix=prompt_suffix
            )
        if attributes or memory_prefix or memory_suffix:
            controls = "" if not attributes else "\n#### Controls:\n{}".format("\n".join(attributes))
            if memory_prefix:
                memory_prefix = memory_prefix + "\n"
            if memory_suffix:
                memory_suffix = "\n" + memory_suffix
            self.memory_template = full_memory_template.format(
                controls=controls,
                prefix=memory_prefix,
                suffix=memory_suffix
            )
