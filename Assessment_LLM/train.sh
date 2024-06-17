train_data_file="../Datasets/train.jsonl"
eval_data_file="../Datasets/valid.jsonl"


if [ $1 = "CodeLlama" ]; then
    model_name_or_path=""
elif [ $1 = "DeepSeek-Coder" ]; then
    model_name_or_path=""
elif [ $1 = "StarCoder" ]; then
    model_name_or_path=""
elif [ $1 = "WizardCoder" ]; then
    model_name_or_path=""
elif [ $1 = "Mistral" ]; then
    model_name_or_path=""
elif [ $1 = "Phi" ]; then
    model_name_or_path=""
fi


python run.py \
    --output_dir ./saved_models/$1 \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --train_data_file $train_data_file \
    --eval_data_file $eval_data_file \
    --num_train_epochs 20 \
    --max_length 512 \
    --train_batch_size 4 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --seed 42 2>&1