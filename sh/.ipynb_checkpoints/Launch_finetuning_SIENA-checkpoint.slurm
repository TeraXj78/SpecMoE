#!/bin/bash

#SBATCH --output=/lustre/fswork/projects/rech/jhu/uav28dl/PART_3/OAR_Training/Exp_SSL_January_26/OAR.CoSupF_PreTrain_Exp10_v1__TUEG_v8_db2.out
#SBATCH --error=/lustre/fswork/projects/rech/jhu/uav28dl/PART_3/OAR_Training/Exp_SSL_January_26/OAR.CoSupF_PreTrain_Exp10_v1__TUEG_v8_db2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu_p2
#SBATCH --hint=nomultithread
#SBATCH --time=90:00:00
#SBATCH --qos=qos_gpu-t4
##SBATCH -C v100-32g
##SBATCH -C a100
#SBATCH -A jhu@v100


module load pytorch-gpu/py3/2.6.0


seed=41


Common_path="/lustre/fswork/projects/rech/jhu/uav28dl/PART_3/SpecMoE"

config_path="$Common_path/Configs/Finetuning_config.json"


gpu=0,1,2



python "$Common_path/Finetuning_main.py" \
    --project_name "Self_Sup_Learning" \
    --Experiment_name "Finetuning_SIENA" \
    --dataset_name "SIENA" \
    --signals_path "/lustre/fswork/projects/rech/jhu/uav28dl/PART_3/Data/SIENA/processed" \
    --batch_size 64 \
    --trained_model_path_1 "/lustre/fswork/projects/rech/jhu/uav28dl/saved/Exp_SSL_January_26/28_01_2026_07_56_47_TestAnimals_0_1_CoSupF_PreTrain_Exp10_v1__TUEG_v7_db2_resume_2_No_WDcay/best_model.pth" \
    --trained_model_path_2 "/lustre/fswork/projects/rech/jhu/uav28dl/saved/Exp_SSL_January_26/27_01_2026_05_46_32_TestAnimals_0_1_CoSupF_PreTrain_Exp10_v1__TUEG_v8_db2_resume/best_model.pth" \
    --trained_model_path_3 "/lustre/fswork/projects/rech/jhu/uav28dl/saved/Exp_SSL_January_26/27_01_2026_06_04_28_TestAnimals_0_1_CoSupF_PreTrain_Exp10_v1__TUEG_v9_db2_resume/best_model.pth" \
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











