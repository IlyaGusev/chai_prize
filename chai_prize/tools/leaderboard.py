import fire
import pandas as pd
import chai_guanaco as chai
from chai_guanaco.login_cli import auto_authenticate
from chai_guanaco.metrics import get_submission_metrics


@auto_authenticate
def get_my_submissions(developer_key):
    submission_ids = chai.get_my_submissions()
    leaderboard = []
    for submission_id in submission_ids:
        metrics = get_submission_metrics(submission_id, developer_key)
        leaderboard.append({'submission_id': submission_id, **metrics})
    return pd.DataFrame(leaderboard)


def main(min_feedback_count: int = 100):
    df = get_my_submissions()
    pd.set_option('display.max_columns', 100)
    df = df[df["total_feedback_count"] > min_feedback_count]
    print(df)


if __name__ == "__main__":
    fire.Fire(main)
