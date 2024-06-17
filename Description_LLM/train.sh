train_data_file="../Datasets/train.jsonl"
valid_data_file="../Datasets/valid.jsonl"
GRADIENT_CHECKPOINTING=true

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
    GRADIENT_CHECKPOINTING=false
fi


python run.py \
    --model_name_or_path $model_name_or_path \
    --train_data_path $train_data_file \
    --valid_data_path $valid_data_file \
    --output_dir ./saved_models/$1 \
    --num_train_epochs 10 \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 10000 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --report_to "tensorboard" \
    --load_best_model_at_end True \
    --bf16 False