import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from chai_prize.conversation import Conversation


class ChatDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        templates_path: str,
        only_target_loss: bool = True,
        labels_pad_token_id: int = -100,
    ):
        self.templates_path = templates_path
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.only_target_loss = only_target_loss
        self.labels_pad_token_id = labels_pad_token_id
        self.is_printed = False

        self.records = []
        for record in tqdm(original_records):
            tensors = self.convert_record(record)
            if tensors is None:
                continue
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def get_tokens(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )["input_ids"]

    def convert_record(self, record):
        conversation = Conversation.from_template(self.templates_path, char_name=record["char_name"])
        conversation.expand(record["messages"])

        bot_messages = []
        full_text = ""
        for message, role in conversation.iter_messages():
            new_text = full_text + message
            new_input_ids = self.get_tokens(new_text)
            if len(new_input_ids) > self.max_tokens_count - 2:
                break
            full_text = new_text
            if role == Conversation.BOT_ROLE:
                bot_messages.append(message)

        input_ids = self.get_tokens(full_text)

        if self.only_target_loss:
            labels = [self.labels_pad_token_id for _ in range(len(input_ids))]
            prev_idx = 0
            for message in bot_messages:
                message_tokens = self.get_tokens("\n" + message)
                message_tokens = message_tokens[2:]
                tokens_count = len(message_tokens)
                message_found = False
                for idx in range(prev_idx, len(labels) - tokens_count + 1):
                    if input_ids[idx: idx + tokens_count] == message_tokens:
                        labels[idx: idx + tokens_count] = message_tokens
                        message_found = True
                        prev_idx = idx + tokens_count
                        break
                assert message_found
        else:
            labels = input_ids[:]

        if len(set(labels)) <= 2:
            return None

        if input_ids[0] != self.tokenizer.bos_token_id and random.random() < 0.5:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            labels.insert(0, self.labels_pad_token_id)

        if not self.is_printed:
            print("Full prompt:", self.tokenizer.decode(input_ids, skip_special_tokens=False))
            print(input_ids)
            print(labels)
            self.is_printed = True

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.new_ones(input_ids.size())
        assert input_ids.size(0) == labels.size(0) == attention_mask.size(0)
        assert input_ids.size(0) <= self.max_tokens_count

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
