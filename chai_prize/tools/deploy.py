import random
import time
from datetime import datetime

import fire
import wandb
import chai_guanaco as chai
from chai_guanaco.metrics import FeedbackMetrics
from chai_guanaco.login_cli import auto_authenticate
from chai_guanaco.feedback import _get_latest_feedback

from chai_prize.tools.submit import submit


@auto_authenticate
def get_feedback(submission_id: str, developer_key=None):
    return _get_latest_feedback(submission_id, developer_key)


def get_submission_metrics(submission_id):
    feedback = get_feedback(submission_id)
    feedback_metrics = FeedbackMetrics(feedback.raw_data)
    metrics = {}
    if len(feedback_metrics.convo_metrics) > 5:
        metrics = {
            'mcl': feedback_metrics.mcl,
            'thumbs_up_ratio': feedback_metrics.thumbs_up_ratio,
            'repetition': feedback_metrics.repetition_score,
            'total_feedback_count': feedback_metrics.total_feedback_count,
            'user_writing_speed': feedback_metrics.user_writing_speed,
        }
    return metrics


def iterate_prev_submissions(submission_id):
    base_name = "_".join(submission_id.split("_")[:-1])
    version = int(submission_id.split("_")[-1][1:])
    for i in range(1, version + 3):
        prev_submission_id = f"{base_name}_v{i}"
        metrics = get_submission_metrics(prev_submission_id)
        yield prev_submission_id, metrics


def get_prev_submissions_max_feedback_count(submission_id):
    total_feedback_count = 0
    for prev_submission_id, metrics in iterate_prev_submissions(submission_id):
        if metrics:
            total_feedback_count = max(metrics["total_feedback_count"], total_feedback_count)
    return total_feedback_count


def compare_metrics(metrics1, metrics2):
    score1 = metrics1["thumbs_up_ratio"] * (2.3 - metrics1["user_writing_speed"])
    score2 = metrics2["thumbs_up_ratio"] * (2.3 - metrics2["user_writing_speed"])
    return score1 < score2


def get_best_past_metrics(submission_id):
    best_metrics = None
    total_feedback_count = 0
    for prev_submission_id, metrics in iterate_prev_submissions(submission_id):
        if metrics and metrics["total_feedback_count"] > total_feedback_count:
            if metrics["total_feedback_count"] >= 150:
                best_metrics = metrics
                total_feedback_count = max(metrics["total_feedback_count"], total_feedback_count)
    return best_metrics


def deploy(
    model_list: str,
    reward_url: str = None,
    thumbs_up_threshold: int = 0.7,
    user_writing_speed_threshold: float = 2.8,
    reject_feedback_count: int = 120,
    accept_feedback_count: int = 150,
    interval: int = 30,
    current_submission_id: str = None,
    current_chosen_model: str = None,
    current_wandb_id: str = None,
    min_top_p: float = 0.8,
    max_top_p: float = 1.0,
    min_top_k: int = 20,
    max_top_k: int = 50,
    min_temperature: float = 0.9,
    max_temperature: float = 1.1,
    min_frequency_penalty: float = 0.0,
    max_frequency_penalty: float = 0.2,
    max_input_tokens: int = 2048,
    best_of: int = 4,
    use_attributes: bool = False,
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    memory_prefix: str = "",
    memory_suffix: str = "",
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
                reward_url=reward_url,
                use_attributes=use_attributes,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                memory_prefix=memory_prefix,
                memory_suffix=memory_suffix,
                top_p=top_p,
                top_k=top_k,
                max_input_tokens=max_input_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                best_of=best_of
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
            if current_wandb_id:
                wandb.init(project="chai_prize", name=submission_id, id=current_wandb_id, resume="must")
            else:
                wandb.init(project="chai_prize", name=submission_id)

        while True:
            print("Submission ID:", submission_id)

            prev_feedback_count = get_prev_submissions_max_feedback_count(submission_id)
            min_feedback_counts[chosen_model] = max(min_feedback_counts[chosen_model], prev_feedback_count)
            print("Prev feedback count:", prev_feedback_count)

            past_metrics = None
            if chosen_model in final_metrics:
                past_metrics = final_metrics[chosen_model]
            true_past_metrics = get_best_past_metrics(submission_id)
            if true_past_metrics:
                if past_metrics is None:
                    final_metrics[chosen_model] = true_past_metrics
                elif compare_metrics(past_metrics, true_past_metrics):
                    final_metrics[chosen_model] = true_past_metrics
                print("Prev best metrics:", final_metrics[chosen_model])

            metrics = get_submission_metrics(submission_id)
            print("Metrics:", metrics)
            total_feedback_count = metrics.get("total_feedback_count", 0)
            wandb.log(metrics, step=total_feedback_count)
            if total_feedback_count == 0:
                time.sleep(interval)
                continue

            user_writing_speed = metrics["user_writing_speed"]
            thumbs_up_ratio = metrics["thumbs_up_ratio"]

            # Early stopping
            is_bad_thumbs_up = thumbs_up_ratio < thumbs_up_threshold
            is_bad_user_writing_speed = user_writing_speed > user_writing_speed_threshold
            is_bad = is_bad_thumbs_up or is_bad_user_writing_speed
            if total_feedback_count > reject_feedback_count and is_bad:
                print("Stopping because of bad metrics!")
                print(f"Feedback count: {total_feedback_count}")
                chai.deactivate_model(submission_id)
                time.sleep(interval)
                wandb.finish()
                break

            # Early stopping because of bad metrics compared to prev model
            if chosen_model in final_metrics and total_feedback_count > min_feedback_counts[chosen_model] - 30:
                past_metrics = final_metrics[chosen_model]
                if compare_metrics(metrics, past_metrics):
                    chai.deactivate_model(submission_id)
                    print("Stopping because of bad metrics!")
                    print("Past metrics:", str(past_metrics))
                    print(f"Feedback count: {total_feedback_count}")
                    time.sleep(interval)
                    wandb.finish()
                    break

            # Normal finish
            if total_feedback_count > min_feedback_counts[chosen_model]:
                min_feedback_counts[chosen_model] = total_feedback_count + 10
                final_metrics[chosen_model] = metrics
                print("Normal finish! Wow! Check your leaderboard")
                chai.deactivate_model(submission_id)
                time.sleep(interval)
                wandb.finish()
                break

            print()
            print(f"Sleeping {interval} seconds")
            time.sleep(interval)


if __name__ == "__main__":
    fire.Fire(deploy)
