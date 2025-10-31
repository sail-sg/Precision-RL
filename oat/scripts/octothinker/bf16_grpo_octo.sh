# Notes -----------------------------------------------
# We use Dr. GRPO by default as the unbiased policy optimization,
# configured by `--critic_type drgrpo`.


# Hyperparameter ---------------------------------------
GPUS=8
BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=128
BATCH_SIZE_PER_DEVICE=1
ROLLOUT_PER_PROMPT=8
MODEL=OctoThinker/OctoThinker-3B-Hybrid-Base
PROMPT_TEMPLATE=r1
DATASET=./data/train/math_12k

python main.py \
    --critic_type drgrpo \
    --gpus $GPUS \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.45 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --no-fp16 \
    --tis_c None \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --oracle_type reward \
    --oracle math \
    --pretrain $MODEL \
    --prompt_template $PROMPT_TEMPLATE \
    --zero-stage 2 \
    --ref_offload \
    --prompt_data $DATASET \
    --train_split train \
    --input_key problem \
    --output_key answer \
    --max-train 9999999 \
    --num_prompt_epoch 2000 \
    --prompt_max_length 1024 \
    --num_samples $ROLLOUT_PER_PROMPT \
    --temperature 1 \
    --top_p 1 \
    --generate_max_length 3000 \
    --max_model_len 4096 \
    --save_steps -1 \
    --train_batch_size $BATCH_SIZE \
    --train_batch_size_per_device $BATCH_SIZE_PER_DEVICE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_batch_size_per_device $(( $ROLLOUT_BATCH_SIZE / $GPUS )) \
    --pi_buffer_maxlen_per_device $(( $ROLLOUT_BATCH_SIZE / $GPUS * $ROLLOUT_PER_PROMPT)) \
    --eval_batch_size 63 \
    --eval_steps 50 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_n 1 \
    --test_split math \
    --eval_generate_max_length 3000 \
    --eval_data ./data/evaluation_suite \
    --eval_input_key input \
    --use-wb \
    --wb_project precision-rl \
    --wb-run-name part6-octo-bf16-grpo
