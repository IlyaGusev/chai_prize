import fire
import chaiverse as chai


def count_user_messages(conversation):
    if "Anonymous user:" in conversation:
        return conversation.count("Anonymous user:")
    return conversation.count("\nXXXXXXXXXXXXXX:")


def show_feedbacks(submission_id: str, user_ids: str = None, public_only: bool = False):
    df = chai.get_feedback(submission_id).df
    if user_ids:
        if isinstance(user_ids, str):
            user_ids = [user_ids]
        records = df[df["user_id"].isin(user_ids)].to_dict(orient="records")
        for record in records:
            print(record["feedback"])
            print()
            print(record["conversation"])
            print("============")
        return
    df = df.drop(["conversation_id", "model_name", "bot_id"], axis=1)
    if public_only:
        df = df[df["public"] == True]
    df["conv_length"] = df["conversation"].apply(lambda x: len(x))
    df["deleted_count"] = df["conversation"].apply(lambda x: x.count("deleted"))
    df["message_count"] = df["conversation"].apply(lambda x: x.count(":"))
    df["feedback"] = df["feedback"].apply(lambda x: x[:50])
    df["user_message_count"] = df["conversation"].apply(count_user_messages)
    df = df.drop(["conversation"], axis=1)
    print(df.to_string())


if __name__ == "__main__":
    fire.Fire(show_feedbacks)
