import sys
from chai_prize.create_set import process_chai
from chai_prize.util.io import write_jsonl

records = process_chai(
    max_length=30000, sample_rate=0.1,
    min_user_engagement_heuristics_score=10.0, only_thumbs_up=True,
    only_good_feedback=True, only_whitelist=True, min_messages=4
)

write_jsonl(records, sys.argv[1])
