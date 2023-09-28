export NCCL_P2P_DISABLE=1

hostfile=""
deepspeed --hostfile=$hostfile tools/tts/fine-tune.py \
    --deepspeed tools/tts/ds_config.json \
    --report_to "tensorboard" \
    --data_path "fishaudio/libritts-r-tokenized" \
    --model_name_or_path "checkpoints/baichuan2-7b-base-extend" \
    --output_dir "results" \
    --model_max_length 2048 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 1000 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --use_lora False \
    --bf16 True \
    --tf32 True
