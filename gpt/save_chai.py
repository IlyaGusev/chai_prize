import sys
from chai_prize.create_set import process_chai
from chai_prize.util.io import write_jsonl

records = process_chai(min_user_engagement=20.0, max_length=20000, sample_rate=0.15)
write_jsonl(records, sys.argv[1])
