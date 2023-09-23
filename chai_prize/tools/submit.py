import json
import fire
import chai_guanaco as chai
from chai_prize.formatter import CustomFormatterV3


def submit(model_url):
    generation_params = {
        "frequency_penalty": 0.5,
        "presence_penalty": 0.0,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "max_input_tokens": 2048,
        "stopping_words": ['\n']
    }

    model_name = model_url.split("/")[-1]
    submission_parameters = {
        "model_repo": model_url,
        "generation_params": generation_params,
        "model_name": model_name,
        "formatter": CustomFormatterV3()
    }

    submitter = chai.ModelSubmitter()
    submission_id = submitter.submit(submission_parameters)
    with open(f"submissions/{submission_id}.json", "w") as w:
        json.dump({
            "model_url": model_url,
            "model_name": model_name,
            "generation_params": generation_params,
            "submission_id": submission_id
        }, w, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(submit)
