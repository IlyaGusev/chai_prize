import fire

from chai_prize.conversation import Conversation
from chai_prize.util.load import load_saiga
from chai_prize.util.generate import generate


def interact(model_name, template_path):
    model, tokenizer, generation_config = load_saiga(model_name)
    char_name = "Makise Kurisu"
    conversation = Conversation.from_template(template_path, char_name=char_name)
    system_message = "You are a genius human girl from Steins;Gate universe"
    attributes = [
        "Verbosity: low",
        "Actions: many",
        "Creativity: high",
        "Capriciousness: low",
        "Fragility: low"
    ]
    controls = "\n#### Controls:\n{}".format("\n".join(attributes))
    system_message += controls
    conversation.add_system_message(system_message)
    conversation.add_prompt_message("")
    conversation.add_bot_message("Hi there! I'm Makisu, nice to meet you! *winks*")
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
            generation_config=generation_config,
            eos_token_id=13
        )[0]
        conversation.add_bot_message(output)
        print(f"{char_name}:", output)


if __name__ == "__main__":
    fire.Fire(interact)
