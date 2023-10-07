import json
import fire
from datetime import datetime
import chai_guanaco as chai
from chai_prize.formatter import CustomFormatterV3


def submit(model_url, **kwargs):
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
    submission_parameters = {
        "model_repo": model_url,
        "generation_params": generation_params,
        "model_name": model_name,
        "formatter": CustomFormatterV3()
    }

    submitter = chai.ModelSubmitter()
    submission_id = submitter.submit(submission_parameters)
    return submission_id.strip(), submission_parameters


if __name__ == "__main__":
    fire.Fire(submit)
