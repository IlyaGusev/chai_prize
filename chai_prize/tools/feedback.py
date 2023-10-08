import sys
from collections import Counter
import chai_guanaco as chai

submission_id = sys.argv[1]
model_feedback = chai.get_feedback(submission_id)
df = model_feedback.df
#df = df[df["public"] == True]
df = df.drop(["conversation_id", "model_name", "bot_id", "public"], axis=1)
print("user_id\tcount")
for user_id, count in Counter(df["user_id"].tolist()).most_common():
    print(str(user_id) + "\t" + str(count))
df["conv_length"] = df["conversation"].apply(lambda x: len(x))
df["deleted_count"] = df["conversation"].apply(lambda x: x.count("deleted"))
df["message_count"] = df["conversation"].apply(lambda x: x.count(":"))
#df = df.drop(["conversation"], axis=1)
#df["user_message_count"] = df["conversation"].apply(lambda x: x.count("Anonymous user"))

#user_id = "SKRM8YAgYRUEB20yUvqPzsOThFi2"
#user_id = "pGOgQoBgH1MRnQNsycIDoCt447S2"
#df = df[df["user_id"] == user_id]
df = df.drop(["conversation"], axis=1)
print(df.to_string())
#print(df["conversation"].tolist())
#print(df["thumbs_up"].sum())
