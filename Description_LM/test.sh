test_data_file="../Datasets/test.jsonl"


if [ $1 = "CodeBERT" ]; then
    model_name_or_path="microsoft/codebert-base"
elif [ $1 = "GraphCodeBERT" ]; then
    model_name_or_path="microsoft/graphcodebert-base"
elif [ $1 = "UniXcoder" ]; then
    model_name_or_path="microsoft/unixcoder-base-nine"
fi


python run.py \
	--do_test\
	--model_name_or_path $model_name_or_path \
	--test_filename $test_data_file \
	--output_dir saved_models/$1 \
	--max_source_length 256 \
	--max_target_length 200 \
	--beam_size 10 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 2>&1