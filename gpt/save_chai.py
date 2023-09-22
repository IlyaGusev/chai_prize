import sys
from chai_prize.create_set import process_pos
from chai_prize.util.io import write_jsonl

records = process_pos(min_user_engagement=20.0, max_length=10000)
write_jsonl(records, sys.argv[1])
