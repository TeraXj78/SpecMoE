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

config_path="$Common_path/Configs/Pretraining_config.json"


n_gpus=8



python "$Common_path/Pretraining_main.py" \
    --project_name "Self_Sup_Learning" \
    --Experiment_name "Pretraining_Expert_1" \
    --Sample_Start 0 \
    --Sample_End 50 \
    --wandb_status "OFF" \
    --n_gpus $n_gpus \
    --config $config_path \
    --seed $seed \
    --testmode "NO" \
    --epochs 100 \
    --batch_size 4 \
    --Accumulation_Grad 4 \
    --lr 1e-3 \
    --weight_decay 5e-2 \
    --clip_value 5.0 \
    --lr_scheduler "CosineAnnealingLR" \
    --need_stft_mask "True" \
    --nb_heads 8














