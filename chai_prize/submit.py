import chai_guanaco as chai
from chai_prize.formatter import RPRFormatter

model_url = "IlyaGusev/rolecuna_13b_v1"

generation_params = {
    "max_new_tokens": 2048,
    "no_repeat_ngram_size": 10,
    "repetition_penalty": 1.13,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 50,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2
}
submission_parameters = {
    "model_repo": model_url,
    "generation_params": generation_params,
    "model_name": "rolecuna_13b_v1",
    "formatter": RPRFormatter()
}

submitter = chai.ModelSubmitter()
submission_id = submitter.submit(submission_parameters)
