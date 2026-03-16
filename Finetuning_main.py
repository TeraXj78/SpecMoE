import os
import argparse
import collections
import numpy as np
import random

import Utils.loss as module_loss
import Utils.metrics as module_metric


from collections import OrderedDict

# Local imports
from Data_loaders.Finetuning_data_loader import data_generator_np
from Model_architectures.Finetuning_model import CoSup_UNet_SSL  
from Model_architectures.Finetuning_model import SSL_3Expert_GatedFusion  

from Trainers.Finetuning_trainer import Trainer
from Utils.Positional_Encoding import RotaryPE             
from Utils.parse_config import ConfigParser
from Utils.utils import delete_directory



import torch
import torch.nn as nn

import time

start = time.time()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) #torch.numel() ==> total number of elts in a tensor

def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def load_ssl_checkpoint(path, nb_heads):
    ssl = CoSup_UNet_SSL(nb_attn_heads_for_SSL=nb_heads)
    ckpt = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    new_sd = OrderedDict()
    for k, v in sd.items():
        name = k[7:] if k.startswith("module.") else k
        new_sd[name] = v

    ssl.load_state_dict(new_sd, strict=True)
    return ssl


def main(params):

    print(f"%%%%%%%%%  SEED = {params.seed} %%%%%%%%% \n")
    random.seed(params.seed)
    os.environ["PYTHONHASHSEED"] = str(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



    logger = config.get_logger2()
    
    
    # === Weights initialization ===
    if params.use_random_init == "YES":
        print("\n--- Training from SCRATCH (Random Weights) ---\n")
        ssl_1 = CoSup_UNet_SSL(nb_attn_heads_for_SSL=params.nb_heads)
        ssl_2 = CoSup_UNet_SSL(nb_attn_heads_for_SSL=params.nb_heads)
        ssl_3 = CoSup_UNet_SSL(nb_attn_heads_for_SSL=params.nb_heads)
        
        ssl_1.apply(weights_init_normal)
        ssl_2.apply(weights_init_normal)
        ssl_3.apply(weights_init_normal)
    else:
        print("\n--- Using PRETRAINED Weights ---\n")
        ssl_1 = load_ssl_checkpoint(params.trained_model_path_1, params.nb_heads)
        ssl_2 = load_ssl_checkpoint(params.trained_model_path_2, params.nb_heads)
        ssl_3 = load_ssl_checkpoint(params.trained_model_path_3, params.nb_heads)


    

    finetune_model = SSL_3Expert_GatedFusion(
        expert1_ssl=ssl_1,
        expert2_ssl=ssl_2,
        expert3_ssl=ssl_3,
        dim_emb=params.SSL_model_dim,  
        num_classes=params.num_classes,  
        L_pool_mode=params.L_pooling_mode,
        C_pool_mode=params.C_pooling_mode,
        gate_hidden=params.gate_hidden,
        gate_temperature=params.gate_temperature,
        clf_hidden=params.clf_hidden,
        clf_dropout=params.clf_dropout,
        freeze_experts=params.freeze_experts,
        normalize_expert_embeddings=params.normalize_expert_embeddings,
        finetuning_mode=params.Finetuning_mode,
        segment_Len_secs=params.segment_Len_secs,
        sampling_Freq=params.sampling_Freq,
        nb_channels=params.N_Structs
    )
        


    logger.info(finetune_model)
    

    # get function handles of loss and metrics
    criterion = getattr(module_loss, params.loss_type)

    # build optimizer
    if params.Finetuning_mode == 0:
        gate_params = list(finetune_model.gate.parameters())
        clf_params  = list(finetune_model.classifier.parameters())
    elif params.Finetuning_mode == 1:
        gate_LinearT_1_params  = list(finetune_model.gate_LinearT_1.parameters())
        gate_LinearT_2_params  = list(finetune_model.gate_LinearT_2.parameters())
        gate_LinearT_3_params  = list(finetune_model.gate_LinearT_3.parameters())
        clf_params  = list(finetune_model.classifier.parameters())
        
    elif params.Finetuning_mode == 2:
        gate_LinearT_1_params  = list(finetune_model.gate_LinearT_1.parameters())
        gate_LinearT_2_params  = list(finetune_model.gate_LinearT_2.parameters())
        gate_LinearT_3_params  = list(finetune_model.gate_LinearT_3.parameters())
        clf_params  = list(finetune_model.classifier.parameters())
        input_encoding_Net_params  = list(finetune_model.input_encoding_Net.parameters())
        


    expert_params = []
    for e in finetune_model.experts:
        expert_params += list(e.parameters())

    # ---- read optimizer args from config ----
    opt_name = config["optimizer"]["type"]          
    opt_args = dict(config["optimizer"]["args"]) 

    base_lr = opt_args.pop("lr")
    expert_lr = 0.0 

    optimizer_cls = getattr(torch.optim, opt_name)
    if params.Finetuning_mode == 0:
        optimizer = optimizer_cls(
            [
                {"params": gate_params + clf_params, "lr": base_lr},
                {"params": expert_params,           "lr": expert_lr},
            ],
            **opt_args
        )
    elif params.Finetuning_mode == 1:
        optimizer = optimizer_cls(
            [
                {"params": gate_LinearT_1_params+gate_LinearT_2_params+gate_LinearT_3_params+clf_params, "lr": base_lr},
                {"params": expert_params,           "lr": expert_lr},
            ],
            **opt_args
        )
    elif params.Finetuning_mode == 2:
        optimizer = optimizer_cls(
            [
                {"params": gate_LinearT_1_params+gate_LinearT_2_params+gate_LinearT_3_params+clf_params+input_encoding_Net_params, "lr": base_lr},
                {"params": expert_params,           "lr": expert_lr},
            ],
            **opt_args
        )


    scheduler_name = config["scheduler"]["type"]
    scheduler_args = dict(config["scheduler"]["args"])

    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
    scheduler = scheduler_cls(optimizer, **scheduler_args)



    if params.dataset_name == "MACO":
        train_dataset, val_dataset, test_dataset = data_generator_maco(signals_path, params.segment_Len_secs, params.original_sampling_Freq, params.sampling_Freq, params.N_Structs, list_files[0],list_files[1],list_files[2])
    else:
        train_dataset, val_dataset, test_dataset = data_generator_np(training_files_path, validation_files_path, test_files_path, params.segment_Len_secs, params.sampling_Freq,params.seed,params.num_classes,params.dataset_name)


    # class weights: more weight for small classes
    total = sum(params.nb_eegs_per_class)
    class_weights = [total / e for e in params.nb_eegs_per_class]  
    weights_for_each_class = [e/sum(class_weights) for e in class_weights]
    print(f"weights_for_each_class : {weights_for_each_class}")

    
    nb_params = count_parameters(finetune_model)
    logger.metric(f"\nNUMBER OF PARAMETERS = {nb_params}")
    logger.metric(f"Batch_Size = {params.batch_size}")



    
    finetuner = Trainer(finetune_model, 
                        criterion, 
                        optimizer, 
                        scheduler, 
                        params.test_mode, 
                        config,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        nb_gpus,
                        weights_for_each_class,
                        params.batch_size,
                        params.Finetuning_mode,
                        params.dataset_name
                    )

    finetuner.train(params.num_classes)

    
    end = time.time()
    elapsed = (end - start)/60
    print("\n")
    print(f"\nTraining time : {elapsed:.2f} minutes --- Arigato !!!!")



if __name__ == '__main__':
    start = time.time()
    
    ##### ---------------------- Finetuning Parameters ---------------------- ############
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('--test_mode', default="NO", type=str,
                      help='test_mode')
    args.add_argument('--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--seed', default=41, type=int,
                      help='seed')
    #------------------
    args.add_argument('--project_name', type=str, default="Project_i")
    args.add_argument('--Experiment_name', type=str, default="Experiment_i")
    args.add_argument('--Finetuning_mode', type=int, default=0)
    args.add_argument('--use_random_init', type=str, default="NO")
    #------------------
    args.add_argument('--nb_heads', type=int, default=8, help='Number of attention heads per layer')
    args.add_argument('--SSL_model_dim', type=int, default=128, help='Emb dim out of SSL Encoder')
    args.add_argument('--L_pooling_mode', type=str, default='mean')
    args.add_argument('--C_pooling_mode', type=str, default='mean')
    args.add_argument('--gate_hidden', type=int, default=128)
    args.add_argument('--gate_temperature', type=float, default=1.5)
    args.add_argument('--clf_hidden', type=int, default=128)
    args.add_argument('--clf_dropout', type=float, default=0.2)
    args.add_argument('--freeze_experts', type=str, default="YES")
    args.add_argument('--normalize_expert_embeddings', type=str, default="YES")
    #------------------
    args.add_argument('--trained_model_path_1', type=str, default='')
    args.add_argument('--trained_model_path_2', type=str, default='')
    args.add_argument('--trained_model_path_3', type=str, default='')
    args.add_argument('--signals_path', type=str, default='')
    args.add_argument('--batch_size', type=int, default=10)
    #------------------

    #------------------
    args.add_argument('--dataset_name', type=str, default="") 


    params = args.parse_args()

    nb_gpus = len(params.device.split(","))
    
    
    config = ConfigParser.from_args(args, params)
    print(config)


    print(f"\nFinetuning Mode : {params.Finetuning_mode}\n")

    if params.dataset_name == "PhysioNet_MI":
        params.num_classes = 4
        params.N_Structs = 64
        params.segment_Len_secs = 4
        params.sampling_Freq = 200
        params.nb_eegs_per_class = [1593,1557,1581,1569]
        params.loss_type = "weighted_CrossEntropyLoss"

        training_files_path   = params.signals_path# + "train/"
        validation_files_path = params.signals_path# + "val/"
        test_files_path       = params.signals_path# + "test/"

    elif params.dataset_name == "BCIC_2020_3":
        params.num_classes = 5
        params.N_Structs = 64
        params.segment_Len_secs = 3
        params.sampling_Freq = 200
        params.nb_eegs_per_class = [900,900,900,900,900]
        params.loss_type = "weighted_CrossEntropyLoss"

        training_files_path   = params.signals_path# + "train/"
        validation_files_path = params.signals_path# + "val/"
        test_files_path       = params.signals_path# + "test/"

    elif params.dataset_name == "SEED_V":
        params.num_classes = 5
        params.N_Structs = 62
        params.segment_Len_secs = 1
        params.sampling_Freq = 200
        params.nb_eegs_per_class = [2976,2976,2976,2976,2976]
        params.loss_type = "weighted_CrossEntropyLoss"

        training_files_path   = params.signals_path# + "train/"
        validation_files_path = params.signals_path# + "val/"
        test_files_path       = params.signals_path# + "test/"

    elif params.dataset_name == "SEED_VIG":
        params.num_classes = 1
        params.N_Structs = 17
        params.segment_Len_secs = 8
        params.sampling_Freq = 200
        params.nb_eegs_per_class = [1]
        params.loss_type = "regression_Loss"

        training_files_path   = params.signals_path# + "train/"
        validation_files_path = params.signals_path# + "val/"
        test_files_path       = params.signals_path# + "test/"

    elif params.dataset_name == "TUAB":
        params.num_classes = 2
        params.N_Structs = 16
        params.segment_Len_secs = 10
        params.sampling_Freq = 200
        params.nb_eegs_per_class = [2976,2976]
        params.loss_type = "weighted_CrossEntropyLoss"

        training_files_path   = params.signals_path + "/train/"
        validation_files_path = params.signals_path + "/val/"
        test_files_path       = params.signals_path + "/test/"

    elif params.dataset_name == "DA_Pharmaco":
        params.num_classes = 5
        params.N_Structs = 5
        params.segment_Len_secs = 60
        params.sampling_Freq = 200
        params.nb_eegs_per_class = [480,240,480,240,240]
        params.loss_type = "weighted_CrossEntropyLoss"

        training_files_path   = params.signals_path# + "train/"
        validation_files_path = params.signals_path# + "val/"
        test_files_path       = params.signals_path# + "test/"


    elif params.dataset_name == "HMC":
        params.num_classes = 5
        params.N_Structs = 4
        params.segment_Len_secs = 30
        params.sampling_Freq = 200
        params.nb_eegs_per_class = [15695, 10544, 33468, 17000, 14541]
        params.loss_type = "weighted_CrossEntropyLoss"

        training_files_path   = params.signals_path# + "train/"
        validation_files_path = params.signals_path# + "val/"
        test_files_path       = params.signals_path# + "test/"


    elif params.dataset_name == "SIENA":
        params.num_classes = 2
        params.N_Structs = 29
        params.segment_Len_secs = 10
        params.sampling_Freq = 200
        params.nb_eegs_per_class = [35190,398]
        params.loss_type = "weighted_CrossEntropyLoss"

        training_files_path   = params.signals_path# + "train/"
        validation_files_path = params.signals_path# + "val/"
        test_files_path       = params.signals_path# + "test/"


    elif params.dataset_name == "MACO":
        params.num_classes = 5
        params.N_Structs = 2
        params.segment_Len_secs = 60
        params.nb_eegs_per_class = [1,1,1,1,1]
        params.loss_type = "weighted_CrossEntropyLoss"
        params.original_sampling_Freq = 1024
        params.sampling_Freq = 200  # The desired sampling frequency

        training_files_path   = params.signals_path# + "train/"
        validation_files_path = params.signals_path# + "val/"
        test_files_path       = params.signals_path# + "test/"


        class_encoding_dict = config["metadata"]["args"]["class_encoding_dict"]
        signals_path = params.signals_path

        post_treatment_ranges_path = config["metadata"]["args"]["post_treatment_ranges_path"]
        metadata_DF_path = config["metadata"]["args"]["metadata_DF_path"]
        Class_Molec_Ani_dict_path = config["metadata"]["args"]["Class_Molec_Ani_dict_path"]
        Class_Molecule_Dose_dict = config["metadata"]["args"]["Class_Molecule_Dose_dict"]
        Test_Animals_index = config["metadata"]["args"]["Test_Animals_index"]


        list_files = load_folds_data(config, params.N_Structs,class_encoding_dict,signals_path,post_treatment_ranges_path,metadata_DF_path,Class_Molec_Ani_dict_path,Class_Molecule_Dose_dict,Test_Animals_index,params.seed)




    print("\n----------------------------------------")
    print("----------------------------------------")
    print(f"     {params.dataset_name} Finetuning")
    print("----------------------------------------")
    print("----------------------------------------\n")


    main(params)






