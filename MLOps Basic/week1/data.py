import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=64):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self) -> None:
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]
    
    def tokenize_data(self, example):
        tokenized = self.tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)
        return tokenized

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type='torch', columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", 
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True 
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )
