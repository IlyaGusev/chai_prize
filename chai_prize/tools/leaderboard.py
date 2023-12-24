import fire
import pandas as pd
from tqdm import tqdm
import chaiverse as chai
from chaiverse.login_cli import auto_authenticate
from chaiverse.metrics import get_submission_metrics


@auto_authenticate
def fetch_my_submissions(developer_key, submission_ids: str = None):
    if submission_ids is not None:
        submission_ids = submission_ids.split(",")
    else:
        submission_ids = [k for k, v in chai.get_my_submissions().items() if v == "deployed"]
    leaderboard = []
    for submission_id in tqdm(submission_ids):
        metrics = get_submission_metrics(submission_id, developer_key)
        leaderboard.append({'submission_id': submission_id, **metrics})
    return pd.DataFrame(leaderboard)


def main(submission_ids: str = None):
    df = fetch_my_submissions(submission_ids=submission_ids)
    pd.set_option('display.max_columns', 100)
    print(df)


if __name__ == "__main__":
    fire.Fire(main)
