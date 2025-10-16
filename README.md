# progen-finetune
LORA fine-tuning pipeline for ProGen2 models


## 1. Setup Env
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt


## 2. Setup Model 
#### - Clone the official Salesforce repo into a temporary folder
git clone https://github.com/salesforce/progen.git temp_progen_repo

#### - Copy the necessary .py files to your project's root directory, and clean up the temporary repo
cp ./temp_progen_repo/progen2/models/progen/*.py ./
rm -rf temp_progen_repo

#### - Your project root should now contain modeling_progen.py and configuration_progen.py.

chmod +x setup_progen_model.sh
#### - Then, run it with the model size you want. Available sizes include progen2-small, progen2-medium, progen2-large, and progen2-xlarge. (Example for the medium model)
./setup_progen_model.sh progen2-medium

#### - This will create a ./checkpoints/progen2-medium directory containing the model files.


## 3. Setup Data
#### - If your data is compressed (e.g., fasta.gz), decompress it first
gunzip your_data.fasta.gz

#### - Run the preparation script
python3 prepare_fasta.py --input_fasta path/to/your_data.fasta --output_dir prepared_data

## 4. Finetune
#### - It's highly recommended to run this inside a tmux session to prevent disconnection during long training runs.

tmux new -s finetuning
source venv/bin/activate

#### - Finally, run the finetuning job
python3 finetune_progen.py \
    --model="./checkpoints/progen2-medium" \
    --train_file="prepared_data/train.txt" \
    --test_file="prepared_data/test.txt" \
    --output_dir="progen2-medium-finetuned-lora" \
    --use_lora
    
Note: Make sure the --model path matches the model you downloaded in Step 2.

## 5: Inference
After finetuning, your new model adapter is saved (e.g., in progen2-medium-finetuned-lora/best_model). 
#### - Use the sample.py script to generate sequences with it.

python3 sample.py \
    --base_model_path="./checkpoints/progen2-medium" \
    --lora_adapter_path="./progen2-medium-finetuned-lora/best_model/" \
    --num_samples=5

#### - This will load the original progen2-medium model, apply your finetuned adapter on top, and generate 5 new protein sequences.
