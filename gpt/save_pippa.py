import sys
from chai_prize.create_set import process_pippa
from chai_prize.util.io import write_jsonl

records = process_pippa(sample_rate=0.2, min_user_engagement=0.0, max_length=20000)
write_jsonl(records, sys.argv[1])
