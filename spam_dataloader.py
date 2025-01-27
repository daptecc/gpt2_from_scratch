from spam_dataset import SpamDataset

import tiktoken
import torch
from torch.utils.data import DataLoader

tokenizer = tiktoken.get_encoding('gpt2')
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataset = SpamDataset(
    csv_file='sms_spam_collection/train.csv',
    max_length=None,
    tokenizer=tokenizer
)
val_dataset = SpamDataset(
    csv_file='sms_spam_collection/validation.csv',
    max_length=None,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file='sms_spam_collection/test.csv',
    max_length=None,
    tokenizer=tokenizer
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)