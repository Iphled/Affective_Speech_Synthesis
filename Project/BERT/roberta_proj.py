import os
from sys import argv
from argparse import ArgumentParser
from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from transformers import RobertaTokenizerFast, RobertaForCausalLM, RobertaConfig
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

os.makedirs(f"{DIRECTORY}/{LANG}-roberta/tokenizer", exist_ok=True)

dataset = load_dataset("wikimedia/wikipedia", f"20231101.{LANG}", cache_dir=args.cache_dir)
dataset = [ data["text"] for data in dataset["train"] ]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
VOCAB_SIZE = 50_000
tokenizer.train_from_iterator(
    dataset,
    vocab_size=VOCAB_SIZE,
    min_frequency=2, 
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

tokenizer.enable_truncation(max_length=args.context_length)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")))
tokenizer.save_model(f"{DIRECTORY}/{LANG}-roberta/tokenizer")

tokenizer = RobertaTokenizerFast.from_pretrained(f"{DIRECTORY}/{LANG}-roberta/tokenizer", max_len=args.context_length)

model = RobertaForCausalLM(
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
    output_dir=f"{DIRECTORY}/{LANG}-roberta",
    overwrite_output_dir=True,
    save_steps=10_000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model(f"{DIRECTORY}/{LANG}-roberta")
