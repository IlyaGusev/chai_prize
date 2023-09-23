import sys
from chai_prize.create_set import process_pippa
from chai_prize.util.io import write_jsonl

records = process_pippa(min_user_engagement=30.0, max_length=10000)
write_jsonl(records, sys.argv[1])
