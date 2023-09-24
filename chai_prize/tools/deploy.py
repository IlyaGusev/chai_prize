import random
import time
import chai_guanaco as chai
from chai_prize.tools.submit import submit

model_list = ["IlyaGusev/rolecuna_d10_m3", "IlyaGusev/rolecuna_d11_m3", "IlyaGusev/rolemax_d10_m3"]
retry_threshold = 0.15
retry_measure_feedback_count = 30
min_feedback_counts = {
    "IlyaGusev/rolecuna_d10_m3": 150,
    "IlyaGusev/rolecuna_d11_m3": 150,
    "IlyaGusev/rolemax_d10_m3": 150
}
final_metrics = dict()

while True:
    chosen_model = random.choice(model_list)
    submission_id = submit(chosen_model)
    while True:
        time.sleep(60)
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
