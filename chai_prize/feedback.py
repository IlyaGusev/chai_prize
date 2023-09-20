import sys
import chai_guanaco as chai

submission_id = sys.argv[1]
model_feedback = chai.get_feedback(submission_id)
model_feedback.sample()
df = model_feedback.df
df = df.drop(["conversation_id", "user_id", "model_name", "bot_id"], axis=1)
print(df)
print(df["thumbs_up"].sum())
