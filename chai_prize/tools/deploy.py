import random
import time
import fire
import chai_guanaco as chai
from chai_prize.tools.submit import submit
from chai_guanaco.metrics import get_submission_metrics


def deploy(
    model_list: str,
    retry_threshold: float = 0.15,
    retry_measure_feedback_count: int = 30,
    starting_min_feedback_count: int = 150
):
    model_list = model_list.split(",")
    min_feedback_counts = {}
    for model in model_list:
        min_feedback_counts[model] = starting_min_feedback_count
    final_metrics = dict()

    while True:
        chosen_model = random.choice(model_list)
        submission_id = submit(chosen_model)
        while True:
            print("Submission ID:", submission_id)
            metrics = get_submission_metrics(submission_id)
            print("Metrics:", metrics)
            retry_score = metrics["retry_score"]
            thumbs_up_ratio = metrics["thumbs_up_ratio"]
            user_engagement = metrics["user_engagement"]
            total_feedback_count = metrics["total_feedback_count"]

            # Early stopping because of retries
            if total_feedback_count > retry_measure_feedback_count and retry_score > retry_threshold:
                print("Stopping because of retries!")
                print(f"Retry score: {retry_score}")
                print(f"Feedback count: {total_feedback_count}")
                chai.deactivate_model(submission_id)
                break

            # Early stopping because of bad metrics
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
                    break

            # Normal finish
            if total_feedback_count > min_feedback_counts[chosen_model]:
                min_feedback_counts[chosen_model] = total_feedback_count + 5
                final_metrics[chosen_model] = metrics
                print("Normal finish! Wow! Check your leaderboard")
                chai.deactivate_model(submission_id)
                break

            print()
            print("Sleeping 60 seconds")
            time.sleep(60)


if __name__ == "__main__":
    fire.Fire(deploy)
