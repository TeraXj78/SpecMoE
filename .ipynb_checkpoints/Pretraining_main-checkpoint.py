import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import time
from datetime import datetime

# Local imports
from Data_loaders.Pretraining_data_loader import data_generator_np
from Model_architectures.Pretraining_model import CoSup_UNet_SSL  
from Trainers.Pretraining_trainer import Trainer
from Utils.Positional_Encoding import RotaryPE             
from Utils.parse_config import ConfigParser
from Utils.utils import delete_directory


# Clean up cached files
#delete_directory("__pycache__/")


# --------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_normal(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def setup_seed(seed):
    print(f"%%%%%%%%%  SEED = {seed} %%%%%%%%% \n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# --------------------------------------------------------------------------




def main():
    logger = config.get_logger2()

    if params.resume == "NO":
        # === Instantiate SSL Model ===
        model = CoSup_UNet_SSL(
            nb_attn_heads=params.nb_heads,
        )
        model.apply(weights_init_normal)
    elif params.resume == "YES":
        # === Instantiate SSL Model ===
        model = CoSup_UNet_SSL(
            nb_attn_heads=params.nb_heads,
        )
        checkpoint_loaded = torch.load(params.resume_path, weights_only=False)
        state_dict_pretrained_M = checkpoint_loaded["model_state"]  
        params.last_epoch = checkpoint_loaded["epoch"] 
        params.optimizer_state = checkpoint_loaded["optimizer_state"]
        params.scheduler_state = checkpoint_loaded["scheduler_state"]
        
        new_state_dict_pretrained_M = OrderedDict()
        for k, v in state_dict_pretrained_M.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict_pretrained_M[name] = v
        model.load_state_dict(new_state_dict_pretrained_M)
        logger.metric(f"MODEL RESUMED AT EPOCH {params.last_epoch+1}")

    nb_params = count_parameters(model)
    logger.metric(f"NUMBER OF PARAMETERS = {nb_params}")
    logger.metric(f"Batch_Size = {params.batch_size}")


    # === Dataset ===
    train_dataset = data_generator_np(
        signals_path=signals_path,
        segment_Len_Secs=segment_Len_secs,
        original_sampling_Freq=original_sampling_Freq,
        desired_sampling_Freq=desired_sampling_Freq,
        nb_structures=nb_Structures,
        masking=params.need_stft_mask,
        mask_params_list = [n_fft,noverlap,mask_ratio,mask_level,mask_type_ratio, band_bias, band_probs, visualize],
        sample_start = params.Sample_Start,
        sample_end = params.Sample_End,
        seed=params.seed
    )

    data_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )


    # === Trainer ===
    trainer = Trainer(config, params, data_loader, model)
    trainer.train()

    train_dataset.db.close()

    # === Done ===
    end = time.time()
    elapsed = (end - start) / 60
    print(f"\nTraining time : {elapsed:.2f} minutes --- Arigato !!!!")
# --------------------------------------------------------------------------   




if __name__ == '__main__':
    start = time.time()
    args = argparse.ArgumentParser(description='EEG SSL Pretraining')

    # --- Paths & config ---
    args.add_argument('--project_name', type=str, default="Project_i")
    args.add_argument('--Experiment_name', type=str, default="Experiment_i")
    args.add_argument('--config', default="config__.json", type=str, help='config file path')
    args.add_argument('-d', '--n_gpus', default=1, type=int, help='number of GPUs to use')
    args.add_argument('-test', '--testmode', default="NO", type=str)
    args.add_argument('-wdb', '--wandb_status', default="ON", type=str)
    

    # --- Training params ---
    args.add_argument('--seed', type=int, default=41)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--weight_decay', type=float, default=0)
    args.add_argument('--clip_value', type=float, default=0.0)
    args.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')
    args.add_argument('--Accumulation_Grad', type=int, default=1)
    args.add_argument('--resume', default="NO", type=str)
    args.add_argument('--resume_path', default="", type=str)

    args.add_argument('--need_stft_mask', type=str, default="True")
    args.add_argument('--nb_heads', type=int, default=8,
                      help='Number of attention heads per layer')

    args.add_argument('--Sample_Start', type=int, default=0)
    args.add_argument('--Sample_End', type=int, default=1109539)

    
    params = args.parse_args()

    # --- Seed and config ---
    setup_seed(params.seed)
    config = ConfigParser.from_args(args, params)

    print(config)
    # === Extract key metadata ===
    segment_Len_secs = config["data_loader"]["args"]["segment_Len_secs"]
    nb_Structures = config["data_loader"]["args"]["N_Structs"]
    signals_path = config["data_loader"]["args"]["signals_path"]
    original_sampling_Freq = config["data_loader"]["args"]["original_sampling_Freq"]
    desired_sampling_Freq = config["data_loader"]["args"]["desired_sampling_Freq"]

    # === Mask parameters ===
    n_fft = config["mask_params"]["n_fft"]
    noverlap = config["mask_params"]["noverlap"]
    mask_ratio = config["mask_params"]["mask_ratio"]
    mask_level = config["mask_params"]["mask_level"]
    mask_type_ratio = config["mask_params"]["mask_type_ratio"]
    band_bias = config["mask_params"]["band_bias"]
    band_probs = config["mask_params"]["band_probs"]
    visualize = config["mask_params"]["visualize"]


    main()



