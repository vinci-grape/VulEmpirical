# VulEmpirical

## Fine-Tuned Model
We share our fine-tuned LLMs and LMs in [https://pan.baidu.com/s/1W4ue9FqcQOxnFBsbs1c_0A?pwd=m4uy]


## Dataset

The dataset is organized as follows:

[Big-Vul](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset)

After download and unzip data files, you should see the following directory structure:

```
VulEmpirical
├── Datasets
    ├── filtered
        ├── train.jsonl
        ├── valid.jsonl
        ├── test.jsonl
    ├── train.jsonl
    ├── valid.jsonl
    ├── test.jsonl
├── ...
```

## Scripts

1. How to Fine-tune LLMs (e.g., CodeLlama for Vulnerability Detection):

```
cd Detection_LLM

train_data_file="../Datasets/filtered/train.jsonl"
eval_data_file="../Datasets/filtered/test.jsonl"
model_name_or_path="codellama/CodeLlama-7b-Instruct-hf"

python run.py \
    --output_dir ./saved_models \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --num_train_epochs 20 \
    --max_length 512 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --seed 42 2>&1
```

2. How to prompt LLMs by Few-Shot learning (e.g., CodeLlama for Vulnerability Detection):

```
cd Detection_LLM

test_data_file="../Datasets/filtered/test.jsonl"
model_name_or_path="codellama/CodeLlama-7b-Instruct-hf"

python fewshot.py \
    --model_name_or_path $model_name_or_path \
    --test_data_file $test_data_file \
    --max_length 512 \
    --seed 42 2>&1
```

After Fine-Tuning or Prompting, your directory structure should be like the following:
```
VulEmpirical
├── dataset
    ├── Big-Vul
├── fewshot
├── result
    ├── assessment
    ├── description
    ├── detection
    ├── location
output
    ├── CodeLlama
    ├── ...
├── ...
```
