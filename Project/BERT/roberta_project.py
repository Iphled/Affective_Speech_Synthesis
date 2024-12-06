import os
from sys import argv
from argparse import ArgumentParser
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import pandas as pd

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, RobertaConfig
from transformers.trainer import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling


parser = ArgumentParser()
parser.add_argument("--output_directory", type=str)
parser.add_argument("--language", type=str)
parser.add_argument("--context_length", type=int, default=128)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--cache_dir", type=str, default=".")
args, _ = parser.parse_known_args(argv[1:])

DIRECTORY = args.output_directory
LANG = args.language

os.makedirs(f"{DIRECTORY}/Classification/tokenizer", exist_ok=True)

dataset = pd.read_csv("data/BERT_training_data2.csv")
training_set = dataset[:int(len(dataset)/5*4)]
test_set = dataset[int(len(dataset)/5*4):]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
    
# Customize training
VOCAB_SIZE = 50_000
tokenizer.train_from_iterator(
    dataset,
    vocab_size=VOCAB_SIZE,
    min_frequency=2, 
    special_tokens=[

    ])

tokenizer.enable_truncation(max_length=args.context_length)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")))
tokenizer.save_model(f"{DIRECTORY}/Classification/tokenizer")

tokenizer = RobertaTokenizerFast.from_pretrained(f"{DIRECTORY}/Classification/tokenizer", max_len=args.context_length)

model = RobertaForSequenceClassification(
    RobertaConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        is_decoder=True
    ))


train_dataset = tokenizer(dataset, add_special_tokens=True, truncation=True, max_length=args.context_length)["input_ids"]
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    learning_rate=1e-3,
    lr_scheduler_type="cosine",
    warmup_steps=1_000,
    weight_decay=0.1,
    gradient_accumulation_steps=8,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    output_dir=f"{DIRECTORY}/Classification",
    overwrite_output_dir=True,
    save_steps=10_000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_set
)

trainer.train()
trainer.save_model(f"{DIRECTORY}/Classification")
