TRAIN_DATA_PATH="./dataset/Big-Vul/train.jsonl"
VALID_DATA_PATH="./dataset/Big-Vul/valid.jsonl"
GRADIENT_CHECKPOINTING=true

if [ $1 = "codellama" ]; then
    OUTPUT_PATH="./output/$2/CodeLlama-7b-Instruct-hf"
    MODEL_PATH="codellama/CodeLlama-7b-Instruct-hf"
elif [ $1 = "deepseek" ]; then
    OUTPUT_PATH="./output/$2/deepseek-coder-6.7b-instruct"
elif [ $1 = "starcoder" ]; then
    OUTPUT_PATH="./output/$2/starcoderbase-7b"
    MODEL_PATH="bigcode/starcoderbase-7b"
elif [ $1 = "wizardcoder" ]; then
    OUTPUT_PATH="./output/$2/WizardCoder-Python-7B-V1.0"
    MODEL_PATH="WizardLM/WizardCoder-Python-7B-V1.0"
elif [ $1 = "mistral" ]; then
    OUTPUT_PATH="./output/$2/Mistral-7B-Instruct-v0.1"
    MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
elif [ $1 = "phi" ]; then
    OUTPUT_PATH="./output/$2/phi-2"
    MODEL_PATH="microsoft/phi-2"
    GRADIENT_CHECKPOINTING=false
fi

python finetune_open_source_LLM.py \
    --model_name_or_path $MODEL_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --valid_data_path $VALID_DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 2048 \
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
    --load_best_model_at_end False \
    --bf16 True

python inference.py --model $1 --task finetune
