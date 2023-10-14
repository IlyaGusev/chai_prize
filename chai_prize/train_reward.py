import random
import fire
from typing import List, Dict

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from trl import RewardTrainer, RewardConfig

from chai_prize.conversation import Conversation
from chai_prize.util.io import read_jsonl
from chai_prize.util.dl import fix_tokenizer, fix_model


class ChatRewardDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        templates_path: str,
        labels_pad_token_id: int = -100
    ):
        self.templates_path = templates_path
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.labels_pad_token_id = labels_pad_token_id
        self.is_printed = False

        self.records = []
        for record in tqdm(original_records):
            assert record["context"][-1]["role"] == "user"
            chosen_record = {
                "messages": record["context"] + [{"role": "bot", "content": record["chosen_message"]}],
                "char_name": record["char_name"]
            }
            rejected_record = {
                "messages": record["context"] + [{"role": "bot", "content": record["rejected_message"]}],
                "char_name": record["char_name"]
            }
            chosen_tensors = self.convert_record(chosen_record)
            rejected_tensors = self.convert_record(rejected_record)
            if not chosen_tensors or not rejected_tensors:
                continue
            self.records.append({
                "input_ids_chosen": chosen_tensors["input_ids"],
                "attention_mask_chosen": chosen_tensors["attention_mask"],
                "input_ids_rejected": rejected_tensors["input_ids"],
                "attention_mask_rejected": rejected_tensors["attention_mask"]
            })

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

        messages = list(conversation.iter_messages())
        system_messages = messages[:2]
        other_messages = messages[2:]

        def calc_tokens_count(messages):
            full_text = "".join([m for m, _ in messages])
            return len(self.get_tokens(full_text)) + 2

        while calc_tokens_count(system_messages + other_messages) > self.max_tokens_count:
            other_messages = other_messages[1:]
            if len(other_messages) <= 1:
                return None

        messages = system_messages + other_messages
        full_text = "".join([m for m, _ in messages])
        input_ids = self.get_tokens(full_text)

        if input_ids[0] != self.tokenizer.bos_token_id and random.random() < 0.7:
            input_ids.insert(0, self.tokenizer.bos_token_id)

        if not self.is_printed:
            print(input_ids)
            print("Full prompt:", self.tokenizer.decode(input_ids, skip_special_tokens=False))
            self.is_printed = True

        input_ids = torch.LongTensor(input_ids)
        attention_mask = input_ids.new_ones(input_ids.size())
        assert input_ids.size(0) == attention_mask.size(0)
        assert input_ids.size(0) <= self.max_tokens_count

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


def train(
    model_name: str,
    train_path: str,
    eval_path: str,
    output_dir: str,
    templates_path: str,
    max_tokens_count: int
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer, model_config)
    train_records = read_jsonl(train_path)
    train_dataset = ChatRewardDataset(
        train_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        templates_path=templates_path
    )
    eval_records = read_jsonl(eval_path)
    eval_dataset = ChatRewardDataset(
        eval_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        templates_path=templates_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = fix_model(model, tokenizer)
    reward_config = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        gradient_accumulation_steps=128,
        gradient_checkpointing=True,
        learning_rate=3e-4,
        report_to="wandb",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=1,
        eval_steps=5,
        save_steps=5,
        warmup_steps=5,
        evaluation_strategy="steps",
        max_length=max_tokens_count,
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
