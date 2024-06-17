test_data_file="../Datasets/test.jsonl"


if [ $1 = "CodeLlama" ]; then
    model_name_or_path=""
    nl_ids=13
elif [ $1 = "DeepSeek-Coder" ]; then
    model_name_or_path=""
    nl_ids=185
elif [ $1 = "StarCoder" ]; then
    model_name_or_path=""
    nl_ids=203
elif [ $1 = "WizardCoder" ]; then
    model_name_or_path=""
    nl_ids=13
elif [ $1 = "Mistral" ]; then
    model_name_or_path=""
    nl_ids=13
elif [ $1 = "Phi" ]; then
    model_name_or_path=""
    nl_ids=198
fi


python run.py \
    --output_dir ./saved_models/$1 \
    --model_name_or_path $model_name_or_path \
    --do_test \
    --test_data_file $test_data_file \
    --max_length 512 \
    --eval_batch_size 1 \
    --nl_ids $nl_ids \
    --seed 42 2>&1