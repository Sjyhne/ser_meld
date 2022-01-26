export MODEL=wav2vec2-base
export TOKENIZER=wav2vec2-base
export ALPHA=0.1
export LR=1e-5
export ACC=1 # batch size * acc = 8
export WORKER_NUM=1

python ravdess_run.py \
--output_dir=output/tmp \
--cache_dir=cache/ \
--num_train_epochs=100 \
--per_device_train_batch_size="2" \
--per_device_eval_batch_size="2" \
--gradient_accumulation_steps=$ACC \
--alpha $ALPHA \
--dataset_name emotion \
--evaluation_strategy="steps" \
--save_total_limit="1" \
--save_steps="500" \
--eval_steps="500" \
--logging_steps="50" \
--logging_dir="log" \
--do_train \
--do_eval \
--learning_rate=$LR \
--model_name_or_path=facebook/$MODEL \
--tokenizer facebook/$TOKENIZER \
--preprocessing_num_workers=$WORKER_NUM \
--gradient_checkpointing true \
--dataloader_num_workers $WORKER_NUM
# --freeze_feature_extractor \