# prepare_fasta.py
import argparse
from pathlib import Path
from Bio import SeqIO
import random

def main():
    parser = argparse.ArgumentParser(description="Prepare a FASTA file for ProGen2 finetuning by converting it to line-by-line text files.")
    parser.add_argument("--input_fasta", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--output_dir", type=str, default="prepared_data", help="Directory to save the train.txt and test.txt files.")
    parser.add_argument("--train_split_ratio", type=float, default=0.9, help="Fraction of data to use for training (0.0 to 1.0).")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading sequences from {args.input_fasta}...")
    
    processed_sequences = []
    for record in SeqIO.parse(args.input_fasta, "fasta"):
        # ProGen2 requires a '1' at the start and '2' at the end of a sequence.
        seq_str = f"1{str(record.seq)}2"
        processed_sequences.append(seq_str)
        
    print(f"Found and processed {len(processed_sequences)} sequences.")

    random.shuffle(processed_sequences)
    split_index = int(len(processed_sequences) * args.train_split_ratio)
    train_data = processed_sequences[:split_index]
    test_data = processed_sequences[split_index:]

    print(f"Splitting into {len(train_data)} training and {len(test_data)} testing samples.")

    # Write to output files
    with open(output_path / "train.txt", 'w') as f:
        f.write('\n'.join(train_data))
            
    with open(output_path / "test.txt", 'w') as f:
        f.write('\n'.join(test_data))

    print(f"Data successfully saved to '{output_path}/train.txt' and '{output_path}/test.txt'")

if __name__ == "__main__":
    main()