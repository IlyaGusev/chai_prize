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


full_memory_template = "{bot_name}'s Persona: {memory}{controls}\n####\n"


class CustomFormatterV3(PromptFormatter):
    memory_template = full_memory_template.format(controls="")
    prompt_template = "{bot_name}: *winks*\n{prompt}\n<START>\n"
    bot_template = "{bot_name}: {message}\n"
    user_template = "User: {message}\n"
    response_template = "{bot_name}:"

    def __init__(self, *args, attributes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if attributes:
            controls = "\n#### Controls:\n{}".format("\n".join(attributes))
            self.memory_template = full_memory_template.format(controls=controls)
