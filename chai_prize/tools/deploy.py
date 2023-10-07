import random
import time
from datetime import datetime

import fire
import wandb
import chai_guanaco as chai
from chai_prize.tools.submit import submit
from chai_guanaco.metrics import get_submission_metrics


def deploy(
    model_list: str,
    retry_threshold: float = 0.18,
    engagement_threshold: int = 120,
    thumbs_up_threshold: int = 0.65,
    reject_feedback_count: int = 100,
    accept_feedback_count: int = 140,
    interval: int = 30,
    current_submission_id: str = None,
    current_chosen_model: str = None,
    current_wandb_id: str = None,
    min_top_p: float = 0.8,
    max_top_p: float = 1.0,
    min_top_k: int = 35,
    max_top_k: int = 50,
    min_temperature: float = 0.9,
    max_temperature: float = 1.1,
    min_frequency_penalty: float = 0.3,
    max_frequency_penalty: float = 0.5,
):
    model_list = model_list.split(",")
    min_feedback_counts = {}
    for model in model_list:
        min_feedback_counts[model] = accept_feedback_count
    final_metrics = dict()

    while True:
        if current_submission_id is None:
            chosen_model = random.choice(model_list)
            top_p = random.uniform(min_top_p, max_top_p)
            top_k = random.randint(min_top_k, max_top_k)
            temperature = random.uniform(min_temperature, max_temperature)
            frequency_penalty = random.uniform(min_frequency_penalty, max_frequency_penalty)
            submission_id, params = submit(
                chosen_model,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                frequency_penalty=frequency_penalty
            )
            params.pop("formatter")
            params["timestamp"] = int(datetime.now().timestamp())
            wandb.init(project="chai_prize", name=submission_id, config=params)
        else:
            assert current_chosen_model
            submission_id = current_submission_id
            chosen_model = current_chosen_model
            current_submission_id = None
            current_chosen_model = None
            wandb.init(project="chai_prize", name=submission_id, id=current_wandb_id, resume="must")

        while True:
            print("Submission ID:", submission_id)
            metrics = get_submission_metrics(submission_id)

            print("Metrics:", metrics)
            total_feedback_count = metrics.get("total_feedback_count", 0)
            wandb.log(metrics, step=total_feedback_count)
            if total_feedback_count == 0:
                time.sleep(interval)
                continue

            retry_score = metrics["retry_score"]
            thumbs_up_ratio = metrics["thumbs_up_ratio"]
            user_engagement = metrics["user_engagement"]

            # Early stopping
            is_bad_retry = retry_score > retry_threshold
            is_bad_engagement = user_engagement < engagement_threshold
            is_bad_thumbs_up = thumbs_up_ratio < thumbs_up_threshold
            is_bad = is_bad_retry or is_bad_engagement or is_bad_thumbs_up
            if total_feedback_count > reject_feedback_count and is_bad:
                print("Stopping because of bad metrics!")
                print(f"Feedback count: {total_feedback_count}")
                chai.deactivate_model(submission_id)
                time.sleep(interval)
                wandb.finish()
                break

            # Early stopping because of bad metrics compared to prev model
            if chosen_model in final_metrics and total_feedback_count > min_feedback_counts[chosen_model] - 20:
                past_metrics = final_metrics[chosen_model]
                past_thumbs_up_ratio = past_metrics["thumbs_up_ratio"]
                past_user_engagement = past_metrics["user_engagement"]
                past_score = past_thumbs_up_ratio * past_user_engagement
                if thumbs_up_ratio * user_engagement < past_score:
                    chai.deactivate_model(submission_id)
                    print("Stopping because of bad metrics!")
                    print("Past metrics:", str(past_metrics))
                    print(f"Feedback count: {total_feedback_count}")
                    time.sleep(interval)
                    wandb.finish()
                    break

            # Normal finish
            if total_feedback_count > min_feedback_counts[chosen_model]:
                min_feedback_counts[chosen_model] = total_feedback_count + 5
                final_metrics[chosen_model] = metrics
                print("Normal finish! Wow! Check your leaderboard")
                chai.deactivate_model(submission_id)
                time.sleep(interval)
                wandb.finish()
                break

            print()
            print(f"Sleeping {interval} seconds")
            time.sleep(interval)
        break


if __name__ == "__main__":
    fire.Fire(deploy)
