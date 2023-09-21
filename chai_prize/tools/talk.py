import fire
import chai_guanaco as chai


def talk(submission_id, char_name: str = "nerd_girl"):
    chatbot = chai.SubmissionChatbot(submission_id)
    chatbot.chat(char_name, show_model_input=True)


if __name__ == "__main__":
    fire.Fire(talk)
