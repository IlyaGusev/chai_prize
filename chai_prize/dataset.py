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
        max_examples_per_record: int = 1
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
            examples_count = 0
            for tensors in self.convert_record(record):
                if tensors is None:
                    continue
                self.records.append(tensors)
                examples_count += 1
                if examples_count == max_examples_per_record:
                    break

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

    def calc_tokens_count(self, messages):
        full_text = "".join([m["content"] for m in messages])
        return len(self.get_tokens(full_text)) + 2

    def cut_system_messages(self, system_messages):
        system_tokens_limit = (self.max_tokens_count - 150) // 2
        if self.calc_tokens_count(system_messages) > system_tokens_limit:
            system_message = system_messages[0]["content"]
            prompt_message = system_messages[1]["content"]
            system_tokens = self.get_tokens(system_message)
            prompt_tokens = self.get_tokens(prompt_message)
            allowed_prompt_tokens = max(system_tokens_limit - len(system_tokens), 0)
            prompt_tokens = prompt_tokens[:allowed_prompt_tokens]
            prompt_message = self.tokenizer.decode(prompt_tokens)
            system_messages[1]["content"] = prompt_message
            if allowed_prompt_tokens == 0 and len(system_tokens) > system_tokens_limit:
                system_tokens = system_tokens[:system_tokens_limit]
                system_message = self.tokenizer.decode(system_tokens)
                system_messages[0]["content"] = system_message
        return system_messages

    def build_bot_mask(self, input_ids, bot_messages, system_tokens_count):
        labels = [self.labels_pad_token_id for _ in range(len(input_ids))]
        prev_idx = max(system_tokens_count - 2, 0)
        for message in bot_messages:
            message_tokens = self.get_tokens("\n" + message)
            message_tokens = message_tokens[2:]
            tokens_count = len(message_tokens)
            message_found = False
            for idx in range(prev_idx, len(labels) - tokens_count + 2):
                if input_ids[idx: idx + tokens_count] == message_tokens:
                    labels[idx: idx + tokens_count] = message_tokens
                    prev_idx = idx + tokens_count
                    message_found = True
                    break
        assert labels != input_ids
        assert any(l != self.labels_pad_token_id for l in labels)
        assert labels[-1] != self.labels_pad_token_id
        return labels

    def process_conv(self, full_text, bot_messages, system_tokens_count):
        input_ids = self.get_tokens(full_text)

        if self.only_target_loss:
            labels = self.build_bot_mask(input_ids, bot_messages, system_tokens_count)
            if labels is None:
                return None
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
        assert input_ids.size(0) <= self.max_tokens_count, input_ids.size(0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    def convert_record(self, record):
        conversation = Conversation.from_template(self.templates_path, char_name=record["char_name"])
        messages = record["messages"]
        system_messages = messages[:2]
        other_messages = messages[2:]
        system_messages = self.cut_system_messages(system_messages)
        messages = system_messages + other_messages
        conversation.expand(messages)

        all_messages = list(conversation.iter_messages())
        system_text = "".join([message for message, role in all_messages[:2]])
        system_tokens_count = len(self.get_tokens(system_text))
        allowed_conv_tokens_count = self.max_tokens_count - system_tokens_count - 3

        conv_text = ""
        new_text = ""
        bot_messages = list()
        last_prev_bot_message = None
        for message_num, (message, role) in enumerate(all_messages[2:]):
            if role == Conversation.USER_ROLE:
                new_text = conv_text + message
                continue

            new_text = new_text + message
            new_tokens_count = len(self.get_tokens(new_text))
            if new_tokens_count < allowed_conv_tokens_count:
                conv_text = new_text
            else:
                is_scrolled = last_prev_bot_message is None or last_prev_bot_message not in conv_text
                if is_scrolled and bot_messages:
                    last_prev_bot_message = bot_messages[-1]
                    yield self.process_conv(system_text + conv_text, bot_messages, system_tokens_count)
                    bot_messages.clear()

                conv_text = new_text
                conv_tokens = self.get_tokens(conv_text)
                conv_tokens = conv_tokens[-allowed_conv_tokens_count + 1:]
                conv_text = self.tokenizer.decode(conv_tokens)

            assert role == Conversation.BOT_ROLE
            if message_num != 0:
                bot_messages.append(message)

        is_scrolled = last_prev_bot_message is None or last_prev_bot_message not in conv_text
        if is_scrolled and bot_messages:
            yield self.process_conv(system_text + conv_text, bot_messages, system_tokens_count)
