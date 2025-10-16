# sample.py
import torch
import argparse
import os
from peft import PeftModel
from modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
'''
python3 sample.py \
    --base_model_path="./checkpoints/progen2-small" \
    --lora_adapter_path="./progen2-finetuned-lora/best_model/" \
    --num_samples=5
'''

def main():
    parser = argparse.ArgumentParser(description="Generate sequences from a finetuned ProGen2 LoRA model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base ProGen2 checkpoint directory.")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the finetuned LoRA adapter directory.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of sequences to generate.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of the generated sequences.")
    args = parser.parse_args()

    # --- CORRECTED TOKENIZER LOADING ---
    print(f"Loading tokenizer from: {args.base_model_path}")
    tokenizer_file_path = os.path.join(args.base_model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_file_path):
        raise FileNotFoundError(f"tokenizer.json not found in model directory: {args.base_model_path}")
    
    regular_tokenizer = Tokenizer.from_file(tokenizer_file_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=regular_tokenizer)

    # Manually set special tokens if they are missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|pad|>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"

    # --- MODEL LOADING ---
    print(f"Loading base model from: {args.base_model_path}")
    model = ProGenForCausalLM.from_pretrained(args.base_model_path, dtype=torch.bfloat16)

    print(f"Loading and applying LoRA adapter from: {args.lora_adapter_path}")
    model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # The prompt is just the start token '1'
    prompt = "1"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print("\n--- Generating Sequences ---")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            num_return_sequences=args.num_samples,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    for i, output in enumerate(outputs):
        # Decode and clean up the sequence
        decoded_sequence = tokenizer.decode(output, skip_special_tokens=True).replace(" ", "")
        # Print sequence without the start/end '1' and '2' tokens
        print(f"\n> Sample {i+1}:\n{decoded_sequence[1:-1]}")

if __name__ == "__main__":
    main()