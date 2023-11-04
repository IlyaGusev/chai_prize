import json
import random
import fire
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from trl import RewardTrainer

from chai_prize.conversation import Conversation
from chai_prize.util.io import read_jsonl
from chai_prize.util.dl import fix_tokenizer, fix_model
from chai_prize.train import custom_prepare_model_for_int8_training, PeftTrainer, SavePeftModelCallback


class ChatRewardDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        templates_path: str,
        labels_pad_token_id: int = -100,
        sample_rate: float = 1.0
    ):
        self.templates_path = templates_path
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.labels_pad_token_id = labels_pad_token_id
        self.is_printed = False
        self.sample_rate = sample_rate

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue

            if "chosen_messages" in record:
                chosen_record = {
                    "messages": record["chosen_messages"],
                    "char_name": record["char_name"]
                }
                rejected_record = {
                    "messages": record["rejected_messages"],
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
                    "attention_mask_rejected": rejected_tensors["attention_mask"],
                })
            else:
                tensors = self.convert_record(record)
                if not tensors:
                    continue
                self.records.append({
                    "input_ids": tensors["input_ids"],
                    "attention_mask": tensors["attention_mask"],
                    "labels": int(record["target"])
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

        if calc_tokens_count(system_messages) > self.max_tokens_count // 2:
            system_message, _ = system_messages[0]
            prompt_message, _ = system_messages[1]
            system_tokens = self.get_tokens(system_message)
            prompt_tokens = self.get_tokens(prompt_message)
            allowed_prompt_tokens = max(self.max_tokens_count // 2 - len(system_tokens), 0)
            prompt_tokens = prompt_tokens[:allowed_prompt_tokens]
            prompt_message = self.tokenizer.decode(prompt_tokens)
            system_messages[1] = (prompt_message, "prompt")
            if allowed_prompt_tokens == 0 and len(system_tokens) > self.max_tokens_count // 2:
                system_tokens = system_tokens[:self.max_tokens_count // 2]
                system_message = self.tokenizer.decode(system_tokens)
                system_messages[0] = (system_message, "system")

        message_count_estimate = self.max_tokens_count // 2 // 10
        if len(other_messages) > message_count_estimate:
            other_messages = other_messages[-message_count_estimate:]

        while calc_tokens_count(system_messages + other_messages) > self.max_tokens_count:
            other_messages = other_messages[2:]
            if len(other_messages) <= 2:
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


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {
        "eval_acc": sum([int(l) == int(p) for l, p in zip(labels, predictions)]) / len(labels),
        "eval_majority_acc": sum(labels) / len(labels)
    }


def train(
    config_file: str,
    train_path: str,
    eval_path: str,
    output_dir: str,
    sample_rate: float = 1.0
):
    with open(config_file, "r") as r:
        config = json.load(r)

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer, model_config)
    train_records = read_jsonl(train_path)
    is_pairwise = "chosen_messages" in train_records[0]
    max_tokens_count = config["max_tokens_count"]
    templates_path = config["templates_path"]
    train_dataset = ChatRewardDataset(
        train_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        templates_path=templates_path,
        sample_rate=sample_rate
    )
    eval_records = read_jsonl(eval_path)
    eval_dataset = ChatRewardDataset(
        eval_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        templates_path=templates_path,
        sample_rate=sample_rate
    )
    print(train_dataset[0])

    load_in_8bit = bool(config.get("load_in_8bit", False))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=config.get("use_flash_attention_2", True)
    )
    model = fix_model(model, tokenizer, use_resize=False)
    if load_in_8bit:
        model = custom_prepare_model_for_int8_training(model)

    lora_config = config.get("lora")
    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)

    trainer_config = config.get("trainer")
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        **trainer_config
    )
    callbacks = []
    trainer_class = Trainer
    if is_pairwise:
        trainer_class = RewardTrainer
    elif lora_config:
        callbacks = [SavePeftModelCallback]
        trainer_class = PeftTrainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        compute_metrics=None if is_pairwise else compute_metrics
    )

    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
