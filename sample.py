import torch
import argparse
from peft import PeftModel
from modeling_progen import ProGenForCausalLM
from transformers import AutoTokenizer
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

    # Load the base model and tokenizer
    print(f"Loading base model from: {args.base_model_path}")
    model = ProGenForCausalLM.from_pretrained(args.base_model_path, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    # Load the LoRA adapter and merge
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
        decoded_sequence = tokenizer.decode(output, skip_special_tokens=True).replace(" ", "")
        print(f"\n> Sample {i+1}:\n{decoded_sequence[1:-1]}") # Print sequence without start/end tokens

if __name__ == "__main__":
    main()
