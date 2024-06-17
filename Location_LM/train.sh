train_data_file="../Datasets/train.jsonl"
eval_data_file="../Datasets/valid.jsonl"
model_type="roberta"


if [ $1 = "PLBART" ]; then
    model_name_or_path="uclanlp/plbart-base"
    model_type="plbart"
elif [ $1 = "CodeT5p" ]; then
    model_name_or_path="Salesforce/codet5p-220m"
    model_type="codet5"
elif [ $1 = "CodeT5" ]; then
    model_name_or_path="Salesforce/codet5-base"
    model_type="codet5"
elif [ $1 = "BERT" ]; then
    model_name_or_path="bert-base-cased"
    model_type="bert"
elif [ $1 = "RoBERTa" ]; then
    model_name_or_path="roberta-base"
elif [ $1 = "CodeBERT" ]; then
    model_name_or_path="microsoft/codebert-base"
elif [ $1 = "GraphCodeBERT" ]; then
    model_name_or_path="microsoft/graphcodebert-base"
elif [ $1 = "UniXcoder" ]; then
    model_name_or_path="microsoft/unixcoder-base-nine"
fi


python run.py \
    --output_dir ./saved_models/$1 \
    --model_name_or_path $model_name_or_path \
    --model_type $model_type \
    --do_train \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --num_train_epochs 20 \
    --block_size 512 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --seed 42 2>&1