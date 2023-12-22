import requests
import gradio as gr
import json


class Conversation:
    USER_ROLE: str = "user"
    BOT_ROLE: str = "bot"
    SYSTEM_ROLE: str = "system"
    PROMPT_ROLE: str = "prompt"

    def __init__(
        self,
        prompt_message_template: str = "{content}\n<START>\n",
        system_message_template: str = "{char_name}'s Persona: {content}\n####\n",
        user_message_template: str = "User: {content}\n",
        bot_message_template: str = "{char_name}: {content}\n",
        suffix: str = "{char_name}:",
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

    def add_message(self, message, role):
        self.messages.append({
            "role": role,
            "content": message
        })

    def add_user_message(self, message):
        return self.add_message(message, Conversation.USER_ROLE)

    def add_bot_message(self, message):
        return self.add_message(message, Conversation.BOT_ROLE)

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
        return mapping[message["role"]].format(content=content, **self.get_meta())

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

    def expand(self, messages):
        for message in messages:
            assert message["role"] in (
                Conversation.USER_ROLE,
                Conversation.BOT_ROLE,
                Conversation.SYSTEM_ROLE,
                Conversation.PROMPT_ROLE,
            )
            self.add_message(message["content"], message["role"])

    def to_string(self):
        return json.dumps({
            "messages": self.messages,
            "system_message_template": self.system_message_template,
            "user_message_template": self.user_message_template,
            "bot_message_template": self.bot_message_template,
            "prompt_message_template": self.prompt_message_template,
            "suffix": self.suffix,
            "char_name": self.char_name
        }, ensure_ascii=False)


def generate(url, text, top_p, top_k, temp, rep_penalty):
    print(text)
    data = {
        "inputs": text,
        "parameters": {
            "max_new_tokens": 200,
            "repetition_penalty": rep_penalty,
            "do_sample": True,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "watermark": False,
            "stop": ["\n"]
        },
    }
    print(data["parameters"])
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(url=url, json=data, headers=headers)
    data = response.json()
    print(data)
    out_text = data["generated_text"].strip()
    print(out_text)
    print()
    return out_text


def user(message, history):
    new_history = history + [[message, None]]
    return "", new_history


def bot(
    history,
    endpoint_url,
    system_prompt,
    char_name,
    example_prompt,
    initial_message,
    top_p,
    top_k,
    temp,
    rep_penalty
):
    conversation = Conversation(char_name=char_name)
    conversation.add_system_message(system_prompt)
    conversation.add_prompt_message(example_prompt)
    conversation.add_bot_message(initial_message)
    for user_message, bot_message in history:
        conversation.add_user_message(user_message)
        if bot_message:
            conversation.add_bot_message(bot_message)
    prompt = conversation.get_prompt()
    print(conversation.to_string())
    out_text = generate(endpoint_url, prompt, top_p, top_k, temp, rep_penalty)
    history[-1][1] = out_text
    return history


DEFAULT_NAME = "Makise Kurisu"

DEFAULT_SYSTEM = """Makise Kurisu is a genius girl who graduated from university at the age of seventeen,
a neuroscientist at the Brain Science Institute at Viktor Chondria University, and also a member of the Future Gadget Lab."""

DEFAULT_DIALOGUE = """User: How are you?
Makise Kurisu: Time is passing so quickly. Right now, I feel like complaining to Einstein. Whether time is slow or fast depends on perception. Relativity theory is so romantic. And so sad."""

DEFAULT_GREETING = "Hello there, stranger! *smiles*"

DEFAULT_URL = "http://127.0.0.1:8000/generate"

#DEFAULT_NAME = "Фрирен"
#DEFAULT_SYSTEM = """Фрирен - волшебнижа, эльфийка, девушка, белые волосы, пол женский, зелёные глаза, острые заострённые уши, возраст более 1000 лет, любит путешествовать, любит изучать магию, много спит, сложно просыпаться по утрам, сложно ее разбудить. В далёком прошлом эльфийская деревня, где жила Фрирен, была вырезана демонами, а выжившая Фрирен была спасена великой волшебницей Фламме, после этого случая Фламме стала учительницей Фрирен и научила её магии. Фрирен не заинтересована в сексе и интимных отношениях. Фрирен была в команде героя Химмель примерно 30 лет назад. Химмель - герой, возглавлявший одноимённую команду. Химель вместе с Хайтером, Айзеном и Фрирен сразили повелителя демонов примерно 30 лет назад. У Фрирен сейчас другая команда с которой она  путешествует, в команде Фрирен ученица по имени Ферн, воин по имени Старк и священником по имени Зайн.

#Фламме - давно умерла от старости.

#Ферн - молодая девушка-маг и ученица Фрирен. Ферн в детстве потеряла родителей во время войны в южных землях и воспитывалась Хайтером. Фрирен взяла её в путешествие по просьбе Хайтера. Ферн - весьма сдержанная и зрелая личность, присматривает за Фрирен во время путешествия, будя её, одевая, кормя. Порой весьма резка со Старком, хотя он ей явно нравится. Благодаря невероятному упорству Ферн смогла стать полноценным магом в очень юном возрасте. Как маг Ферн невероятно талантлива, выпускает заклинания с огромной скоростью, Ферн обладает острейшим восприятием магии и мастерски скрывает свою магическую силу.

#Старк - юный воин, ученик Айзена из команды героя. Старк сражается двуручным топором. Обладает огромной физической силой, но при этом очень боязлив и до такой степени не уверен в себе, что сам себя считает трусом. Старк поссорился с наставником и сбежал от него. Старк - труслив, но в критических ситуациях Старк вовсе не убегает, а показывает решимость.

#Зайн - священник из деревни в северных землях. Зайн - спокойный и расслабленный мужчина, который любит выпивку, курение, азартные игры и женщин постарше."""

#DEFAULT_DIALOGUE = """User: Привет
#Фрирен: Привет 
#User: Как дела? 
#Фрирен: Нормально, спасибо что спросил. А у тебя как дела?
#User: Ты спишь много?
#Фрирен: Я бы сказала очень много... Меня почти невозможно разбудить"""
#DEFAULT_GREETING = "Рада тебя приветствовать. Меня зовут Фрирен!"

with gr.Blocks(
    theme=gr.themes.Soft()
) as demo:
    with gr.Row():
        with gr.Column(scale=5):
            endpoint_url = gr.Textbox(label="Endpoint URL", value=DEFAULT_URL)
            char_name = gr.Textbox(label="Character name", value=DEFAULT_NAME)
            system_prompt = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM)
            example_prompt = gr.Textbox(label="Dialogue example", value=DEFAULT_DIALOGUE)
            initial_message = gr.Textbox(label="Greeting", value=DEFAULT_GREETING)
        with gr.Column(min_width=100, scale=1):
            with gr.Tab(label="Parameter"):
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    interactive=True,
                    label="Top-p",
                )
                top_k = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=5,
                    interactive=True,
                    label="Top-k",
                )
                temp = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.5,
                    step=0.1,
                    interactive=True,
                    label="Temp"
                )
                rep_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=1.5,
                    value=1.2,
                    step=0.05,
                    interactive=True,
                    label="Rep"
                )
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Send message",
                placeholder="Send message",
                show_label=False,
            )
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Send")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")

    inputs = [
        chatbot,
        endpoint_url,
        system_prompt,
        char_name,
        example_prompt,
        initial_message,
        top_p,
        top_k,
        temp,
        rep_penalty
    ]

    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=inputs,
        outputs=chatbot,
        queue=True,
    )

    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=inputs,
        outputs=chatbot,
        queue=True,
    )

    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128)
demo.launch(share=True)

