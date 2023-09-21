import json
import chai_guanaco as chai
from chai_prize.formatter import CustomFormatterV2

model_url = "IlyaGusev/rolecuna_13b_v4"

generation_params = {
    "frequency_penalty": 1.2,
    "presence_penalty": 0.0,
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 50
}

submission_parameters = {
    "model_repo": model_url,
    "generation_params": generation_params,
    "model_name": "rolecuna_13b_v4",
    "formatter": CustomFormatterV2()
}

submitter = chai.ModelSubmitter()
submission_id = submitter.submit(submission_parameters)
with open(f"submissions/{submission_id}.json", "w") as w:
    json.dump({
        "model_url": model_url,
        "generation_params": generation_params,
        "submission_id": submission_id
    }, w, indent=4, ensure_ascii=False)
