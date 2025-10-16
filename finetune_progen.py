# finetune_final.py
"""
A robust script to finetune ProGen2 models on custom protein sequence data.

This script is designed to work with a local checkpoint of a ProGen2 model and a
prepared dataset of protein sequences stored in simple text files (one sequence per line).
It incorporates modern best practices such as LoRA for memory-efficient finetuning,
dynamic batch padding, and the AdamW optimizer.

Prerequisites:
1. A local ProGen2 model checkpoint directory containing:
   - pytorch_model.bin
   - config.json
   - tokenizer.json
2. Local Python files defining the model architecture:
   - modeling_progen.py
   - configuration_progen.py
3. A training data file (e.g., train.txt).
4. A testing/validation data file (e.g., test.txt).

Example Usage:
    python3 finetune_progen.py \
        --model="./checkpoints/progen2-small" \
        --train_file="prepared_data/train.txt" \
        --test_file="prepared_data/test.txt" \
        --output_dir="progen2-finetuned-lora" \
        --use_lora
"""

import argparse
import os
import logging
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# --- CORRECTED IMPORTS ---
# Use the core `tokenizers` library for reliable loading, and `PreTrainedTokenizerFast`
# to make it compatible with the rest of the transformers ecosystem.
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from modeling_progen import ProGenForCausalLM
from peft import LoraConfig, get_peft_model

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class FastaDataset(Dataset):
    """
    A simple PyTorch Dataset for reading a text file where each line is a protein sequence.
    """
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> str:
        return self.lines[idx]


class PadCollate:
    """
    A collate function to handle dynamic padding of batches. This is essential for
    training efficiently with sequences of varying lengths.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        encoded['labels'] = encoded['input_ids'].clone()
        return encoded


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
    current_epoch: int
) -> float:
    """Performs one full training epoch."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {current_epoch + 1}/{args.epochs} | Training")

    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / args.accumulation_steps
        loss.backward()
        total_loss += loss.item()

        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix({'loss': f"{(total_loss * args.accumulation_steps / (i + 1)):.4f}"})

    return (total_loss * args.accumulation_steps) / len(dataloader)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, args: argparse.Namespace) -> float:
    """Performs evaluation on the validation set."""
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        pbar.set_postfix({'eval_loss': f"{(total_loss / len(pbar)):.4f}"})
        
    return total_loss / len(dataloader)


def main(args: argparse.Namespace):
    """The main function to orchestrate the finetuning process."""
    torch.manual_seed(args.seed)

    # --- 1. Load Tokenizer and Model (Corrected Method) ---
    logger.info(f"Loading tokenizer from local path: {args.model}")
    
    # Load the tokenizer.json file directly using the `tokenizers` library.
    tokenizer_file_path = os.path.join(args.model, "tokenizer.json")
    if not os.path.exists(tokenizer_file_path):
        raise FileNotFoundError(f"tokenizer.json not found in model directory: {args.model}")
        
    regular_tokenizer = Tokenizer.from_file(tokenizer_file_path)
    
    # Wrap it in a `PreTrainedTokenizerFast` to make it fully compatible with transformers.
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=regular_tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|pad|>"  # ProGen uses a specific pad token
        tokenizer.pad_token_id = 0
        
    if tokenizer.eos_token is None:
        # Set an arbitrary end-of-sequence token if not present
        tokenizer.eos_token = "<|endoftext|>"

    logger.info(f"Loading model from local path: {args.model}")
    model = ProGenForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16
    ).to(args.device)
    
    logger.info(f"Model loaded with {model.num_parameters() / 1e6:.2f}M parameters.")

    if args.use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["qkv_proj"],  # <-- This is the correct layer name
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA enabled. Trainable parameters:")
        model.print_trainable_parameters()
    
    # --- 2. Prepare Datasets ---
    train_dataset = FastaDataset(args.train_file)
    test_dataset = FastaDataset(args.test_file)
    
    collate_fn = PadCollate(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn
    )

    # --- 3. Setup Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    training_steps = args.epochs * len(train_dataloader) // args.accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=training_steps
    )

    # --- 4. Training Loop ---
    logger.info("Starting finetuning process...")
    best_eval_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, args, epoch)
        eval_loss = evaluate(model, test_dataloader, args)
        
        logger.info(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            output_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"New best model saved to {output_dir} (Eval Loss: {best_eval_loss:.4f})")

    logger.info("Finetuning finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A robust and modern script for finetuning ProGen2 models.")
    
    parser.add_argument("--model", type=str, required=True, help="Path to the local model checkpoint directory.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data text file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the validation data text file.")
    parser.add_argument("--output_dir", type=str, default="./progen2-finetuned", help="Directory to save the best model checkpoint.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (e.g., 'cuda', 'cpu').")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Steps to accumulate gradients.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of learning rate warmup steps.")
    
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA for memory-efficient finetuning.")
    
    args = parser.parse_args()
    main(args)