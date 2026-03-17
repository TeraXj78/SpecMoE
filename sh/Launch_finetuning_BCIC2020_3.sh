#!/bin/bash

# Environment loading ......

seed=41

Common_path="......../SpecMoE"

gpu=0,1



config_path="$Common_path/Configs/Finetuning_config_BCIC2020_3.json"

python "$Common_path/Finetuning_main.py" \
    --project_name "Self_Sup_Learning" \
    --Experiment_name "Finetuning_BCIC2020_3" \
    --dataset_name "BCIC_2020_3" \
    --signals_path "......../Data/BCIC2020_3/processed" \
    --batch_size 64 \
    --trained_model_path_1 "$Common_path/Model_weights/Pretrained_Expert_1.pth" \
    --trained_model_path_2 "$Common_path/Model_weights/Pretrained_Expert_2.pth" \
    --trained_model_path_3 "$Common_path/Model_weights/Pretrained_Expert_3.pth" \
    --config $config_path \
    --seed $seed \
    --device $gpu \
    --test_mode "NO" \
    --nb_heads 8 \
    --L_pooling_mode "mean" \
    --C_pooling_mode "mean" \
    --SSL_model_dim 128 \
    --gate_hidden 128 \
    --gate_temperature 1.5 \
    --clf_hidden 128 \
    --clf_dropout 0.2 \
    --freeze_experts "YES" \
    --normalize_expert_embeddings "YES" \
    --Finetuning_mode 1











