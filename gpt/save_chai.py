import sys
from collections import Counter
from chai_prize.create_set import process_chai
from chai_prize.util.io import write_jsonl

records = process_chai(
    max_length=30000, sample_rate=1.0,
    min_user_engagement_heuristics_score=10.0, only_thumbs_up=True,
    only_good_feedback=True, only_whitelist=True, min_messages=4
)

cnt = Counter([r["char_name"] for r in records])
for char_name, cnt in cnt.most_common(1000):
    print(char_name, cnt)

#write_jsonl(records, sys.argv[1])
