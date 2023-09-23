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


class CustomFormatterV3(PromptFormatter):
    memory_template = "{bot_name}'s Persona: {memory}\n####\n"
    prompt_template = "{prompt}\n<START>\n"
    bot_template = "{bot_name}: {message}\n"
    user_template = "User: {message}\n"
    response_template = "{bot_name}:"

