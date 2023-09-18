from chai_guanaco.formatters import PromptFormatter


class RPRFormatter(PromptFormatter):
    memory_template = "<s>system\nYou are {bot_name}. {memory}</s>"
    prompt_template = "<s>prompt\n{prompt}</s>"
    bot_template = "<s>bot\n{message}</s>"
    user_template = "<s>user\n{message}</s>"
    response_template = "<s>bot\n"
