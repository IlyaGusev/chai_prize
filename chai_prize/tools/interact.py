import fire

from chai_prize.conversation import Conversation
from chai_prize.util.load import load_saiga
from chai_prize.util.generate import generate


def interact(model_name, template_path):
    model, tokenizer, generation_config = load_saiga(model_name)
    char_name = input("Character name: ")
    conversation = Conversation.from_template(template_path, char_name=char_name)
    system_message = input("System: ")
    conversation.add_system_message(system_message)
    while True:
        user_message = input("User: ")
        if user_message.strip() == "/reset":
            conversation = Conversation.from_template(template_path)
            conversation.add_system_message(system_message)
            print("History reset completed!")
            continue
        conversation.add_user_message(user_message)
        prompt = conversation.get_prompt()
        print(prompt)
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            generation_config=generation_config
        )[0]
        conversation.add_bot_message(output)
        print("Bot:", output)


if __name__ == "__main__":
    fire.Fire(interact)
