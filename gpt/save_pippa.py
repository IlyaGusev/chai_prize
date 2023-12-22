import sys
from chai_prize.create_set import process_pippa
from chai_prize.util.io import write_jsonl

records = process_pippa(
    sample_rate=1.0,
    max_length=20000,
    min_action_heuristics_score=0.6,
    min_messages=6
)
write_jsonl(records, sys.argv[1])
