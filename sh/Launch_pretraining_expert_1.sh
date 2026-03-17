#!/bin/bash

# Environment loading ......

seed=41

Common_path="......../SpecMoE"

n_gpus=8



config_path="$Common_path/Configs/Pretraining_config.json"


python "$Common_path/Pretraining_main.py" \
    --project_name "Self_Sup_Learning" \
    --Experiment_name "Pretraining_Expert_1" \
    --Sample_Start 0 \
    --Sample_End 400000 \
    --wandb_status "OFF" \
    --n_gpus $n_gpus \
    --config $config_path \
    --seed $seed \
    --testmode "NO" \
    --epochs 50 \
    --batch_size 64 \
    --Accumulation_Grad 4 \
    --lr 1e-3 \
    --weight_decay 5e-2 \
    --clip_value 5.0 \
    --lr_scheduler "CosineAnnealingLR" \
    --need_stft_mask "True" \
    --nb_heads 8














