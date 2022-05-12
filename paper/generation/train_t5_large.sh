python /workspace/transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-large \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang pa \
    --source_prefix "paraphrase: " \
    --output_dir /workspace/paraphrase-metrics/generation/t5-large-model \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --train_file /workspace/paraphrase-metrics/datasets/trainlines.json \
    --validation_file /workspace/paraphrase-metrics/datasets/testlines.json \
    --warmup_steps=300 \
    --logging_step=100 \
    --learning_rate=1e-5 \
    --num_train_epochs=10 \
    --gradient_accumulation_steps=2 \
    --max_source_length=64 \
    --max_target_length=64 \
    --predict_with_generate \
    --load_best_model_at_end \
    --logging_strategy=steps \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --save_total_limit=2 

rm -rf /workspace/paraphrase-metrics/generation/t5-large-model/checkpoint-*
