import fire
import chaiverse as chai
from chai_prize.formatter import CustomFormatterV3


class CustomModelSubmitter(chai.ModelSubmitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def submit(self, submission_params):
        submission_params = self._preprocess_submission(submission_params)
        submission_id = self._get_submission_id(submission_params)
        self._print_submission_header(submission_id)
        status = self._wait_for_model_submission(submission_id)
        self._print_submission_result(status)
        self._progress = 0
        return submission_id


def submit(
    model_url: str,
    reward_url: str = None,
    **kwargs
):
    generation_params = {
        "frequency_penalty": 0.45,
        "presence_penalty": 0.0,
        "temperature": 1.0,
        "top_p": 0.85,
        "top_k": 30,
        "max_input_tokens": 2048,
        "stopping_words": ['\n']
    }
    for key, value in kwargs.items():
        generation_params[key] = value

    print(generation_params)
    model_name = model_url.split("/")[-1]
    formatter = CustomFormatterV3()
    formatter = CustomFormatterV3()
    submission_parameters = {
        "model_repo": model_url,
        "generation_params": generation_params,
        "model_name": model_name,
        "formatter": formatter,
    }
    if reward_url:
        submission_parameters["reward_repo"] = reward_url

    print(submission_parameters)
    submitter = CustomModelSubmitter()
    submission_id = submitter.submit(submission_parameters)
    return submission_id.strip(), submission_parameters


if __name__ == "__main__":
    fire.Fire(submit)
