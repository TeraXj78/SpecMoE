import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from numpy import inf
import numpy as np
import pandas as pd
import math
import random
from Utils.metrics import Metrics_Computation, RegressionMetrics
from fastprogress import progress_bar
random.seed(226)
from Utils.Positional_Encoding import RotaryPE

selected_d = {"outs": [], "trg": []}


def stats(x):
    return x.mean().item(), x.std().item(), x.min().item(), x.max().item()



class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, testmode, config, train_dataset, val_dataset, test_dataset, nb_gpus, class_weights, batch_size, finetuning_mode, dataset_name): 
        self.config = config

        self.logger = config.get_logger2()

        self.dataset_name = dataset_name
        self.accumulation_steps = config['trainer'].get('accumulation_steps', 1)
        self.unfreeze_epoch = config['trainer'].get('unfreeze_epoch', 1) 
        self.lr_experts = config['trainer'].get('lr_experts', 1e-4) 
        self.lambda_energy_loss = config['trainer'].get('lambda_energy_loss', 0)


        print(f"Accumulation_steps : {self.accumulation_steps}\n")

        self.device, device_ids = self._prepare_device(nb_gpus)
        self.model = model.to(self.device)
        self.PE_object = RotaryPE(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
            
        self.best_val_f1_score = 0.1     
        self.best_val_rmse = -float("inf")

        self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_from_config = scheduler 

        self.testmode = testmode
        self.class_weights = torch.tensor(class_weights).to(self.device)
        self.batch_size = batch_size
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        cfg_trainer = config['trainer']

        if self.testmode == "YES":
            self.epochs = 1
        else:
            self.epochs = cfg_trainer['epochs']

        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        self.finetuning_mode = finetuning_mode


        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        
        self.dico_metadata_train = {}
        self.dico_metadata_val = {}
        self.dico_metadata_test = {}
        self.r_list = []


    def pad_to_multiple_1d(self, x: torch.Tensor, multiple: int, mode: str = "constant"):
        assert x.ndim == 3, f"Expected [B,C,L], got {x.shape}"
        L_orig = x.shape[-1]
        L_pad = int(math.ceil(L_orig / multiple) * multiple)
        pad_right = L_pad - L_orig
    
        if pad_right == 0:
            return x, 0, L_orig
    
    
        x_pad = F.pad(x, (0, pad_right), mode=mode)
        return x_pad.float(), pad_right, L_orig


    def _train_epoch(self, train_loader, val_loader, test_loader, nb_classes):

        self.len_epoch = len(train_loader)
        self.do_validation = val_loader is not None
        self.lr_scheduler = self.optimizer



        self.selected=0
        nb_chunks = 1
        self.model.train()
        
        val_outs, val_trgs, test_outs, test_trgs = [],[],[],[] 
        if nb_classes == 1:
            NEW_train_outs = torch.tensor([]).to(self.device)
            NEW_train_trgs = torch.tensor([]).to(self.device)
        else:
            NEW_train_outs = torch.tensor([[0]*nb_classes]).to(self.device)
            NEW_train_trgs = torch.tensor([0]).to(self.device)

        mean_train_loss = 0

        self.optimizer.zero_grad()
        batch_idx = 0

        for (data, times, target) in tqdm(train_loader,desc="Train",
                                                    file=sys.stderr,
                                                    mininterval=5,  
                                                    leave=True):
            if self.dataset_name == "MACO":
                data = data.reshape(data.size(0)*data.size(1), data.size(2), data.size(3))
                times = times.reshape(times.size(0)*times.size(1), times.size(2)).unsqueeze(1)
                target = target.reshape(target.size(0)*target.size(1))
            else:
                times = times.reshape(times.size(0), times.size(1)).unsqueeze(1)

            data = data.float().to(self.device)
            
            times = times.float().to(self.device)



            #-----------   Data padding to allow "integer" maxpooling outputs ----------------------
            TOTAL_DOWNSAMPLE = 15 * 4 * 2
            data_padded, pad_right, _ = self.pad_to_multiple_1d(data, multiple=TOTAL_DOWNSAMPLE, mode="constant")
            times_padded, _, _        = self.pad_to_multiple_1d(times, multiple=TOTAL_DOWNSAMPLE, mode="constant")
            #---------------------------------------------------------------------------------------


            if self.finetuning_mode == 0:
                output, w, gate_logits = self.model(data_padded, times_padded, self.PE_object)  
            else:
                output = self.model(data_padded, times_padded, self.PE_object) 

            if nb_classes == 1:
                target = target.to(dtype=torch.float32, device=self.device)
                target = target.view(-1)         
                output = output.squeeze(-1)
                probs = output
            else:
                target = target.to(dtype=torch.long, device=self.device)
                probs = torch.softmax(output, dim=1)

            # Save probs for later analysis
            NEW_train_outs = torch.cat([NEW_train_outs, probs], dim=0)
            NEW_train_trgs = torch.cat([NEW_train_trgs, target], dim=0)

            # Loss on logits
            output_chunks = torch.chunk(output, chunks=nb_chunks, dim=0)
            target_chunks = torch.chunk(target, chunks=nb_chunks, dim=0)

            loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            if self.finetuning_mode == 0:
                for chunck_i in range(nb_chunks):
                    loss_i, entropy_loss_i = self.criterion(
                        output_chunks[chunck_i],
                        target_chunks[chunck_i],
                        self.class_weights,
                        self.device,
                        gate_weights=w,
                        lambda_gate=self.lambda_energy_loss
                    )
                    loss += loss_i
                    entropy_loss += entropy_loss_i
                loss /= nb_chunks
                entropy_loss /= nb_chunks
            else:
                for chunck_i in range(nb_chunks):
                    loss_i = self.criterion(
                        output_chunks[chunck_i],
                        target_chunks[chunck_i],
                        self.class_weights,
                        self.device,
                        gate_weights=None,
                        lambda_gate=0
                    )
                    loss += loss_i
                loss /= nb_chunks
                
            if batch_idx == self.len_epoch-1 and self.finetuning_mode == 0:
                w_mean = w.detach().mean(dim=0)
                self.logger.metric(f"Gate mean weights  : {w_mean.cpu().tolist()}")
                self.logger.metric(f"Entropy_loss-Train : {entropy_loss.detach().clone()*self.lambda_energy_loss}")
            
            raw_loss = loss.detach().clone()
            (loss / self.accumulation_steps).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            mean_train_loss += raw_loss
            batch_idx += 1

        mean_train_loss = mean_train_loss / self.len_epoch

        if (batch_idx) % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

            
        if self.do_validation:
            mean_val_loss, NEW_val_outs, NEW_val_trgs = self._valid_epoch(val_loader, nb_classes)
            mean_test_loss, NEW_test_outs, NEW_test_trgs = self._valid_epoch(test_loader, nb_classes) 

            self.scheduler.step(mean_val_loss)


        if nb_classes == 1:
            return mean_train_loss, mean_val_loss, mean_test_loss, NEW_train_outs, NEW_train_trgs, NEW_val_outs, NEW_val_trgs, NEW_test_outs, NEW_test_trgs
        else:
            return mean_train_loss, mean_val_loss, mean_test_loss, NEW_train_outs[1:,:], NEW_train_trgs[1:], NEW_val_outs, NEW_val_trgs, NEW_test_outs, NEW_test_trgs


    def _valid_epoch(self, loader, nb_classes):

        self.model.eval()
        self.len_val_epoch = len(loader)
        nb_chunks = 1
        with torch.no_grad():
            if nb_classes == 1:
                NEW_outs = torch.tensor([]).to(self.device)
                NEW_trgs = torch.tensor([]).to(self.device)
            else:
                NEW_outs = torch.tensor([[0]*nb_classes]).to(self.device)
                NEW_trgs = torch.tensor([0]).to(self.device)

            mean_val_loss = 0

            for (data, times, target) in tqdm(loader, desc="Val/Test",
                                                file=sys.stderr,
                                                mininterval=5,
                                                leave=True):
                if self.dataset_name == "MACO":
                    data = data.reshape(data.size(0)*data.size(1), data.size(2), data.size(3))
                    times = times.reshape(times.size(0)*times.size(1), times.size(2)).unsqueeze(1)
                    target = target.reshape(target.size(0)*target.size(1))
                else:
                    times = times.reshape(times.size(0), times.size(1)).unsqueeze(1)

                data = data.float().to(self.device)

                
                times = times.float().to(self.device)
                

                #-----------   Data padding to allow "integer" maxpooling outputs ----------------------
                TOTAL_DOWNSAMPLE = 15 * 4 * 2
                data_padded, pad_right, _ = self.pad_to_multiple_1d(data, multiple=TOTAL_DOWNSAMPLE, mode="constant")
                times_padded, _, _        = self.pad_to_multiple_1d(times, multiple=TOTAL_DOWNSAMPLE, mode="constant")
                #---------------------------------------------------------------------------------------

                    
                if self.finetuning_mode == 0:
                    output, w, gate_logits = self.model(data_padded, times_padded, self.PE_object)  
                else:
                    output = self.model(data_padded, times_padded, self.PE_object) 


                if nb_classes == 1:
                    target = target.to(dtype=torch.float32, device=self.device)
                    target = target.view(-1)
                    output = output.squeeze(-1)
                    probs = output
                else:
                    target = target.to(dtype=torch.long, device=self.device)
                    probs = torch.softmax(output, dim=1)

                
                NEW_outs = torch.cat([NEW_outs, probs], dim=0)
                NEW_trgs = torch.cat([NEW_trgs, target], dim=0)

                output_chunks = torch.chunk(output, chunks=nb_chunks, dim=0)
                target_chunks = torch.chunk(target, chunks=nb_chunks, dim=0)

                loss = torch.tensor(0.0, device=self.device)
                entropy_loss = torch.tensor(0.0, device=self.device)
                if self.finetuning_mode == 0:
                    for chunck_i in range(nb_chunks):
                        loss_i, entropy_loss_i = self.criterion(
                            output_chunks[chunck_i],
                            target_chunks[chunck_i],
                            self.class_weights,
                            self.device,
                            gate_weights=w,
                            lambda_gate=self.lambda_energy_loss
                        )
                        loss += loss_i
                        entropy_loss += entropy_loss_i
                    loss /= nb_chunks
                    entropy_loss /= nb_chunks
                else:
                    for chunck_i in range(nb_chunks):
                        loss_i = self.criterion(
                            output_chunks[chunck_i],
                            target_chunks[chunck_i],
                            self.class_weights,
                            self.device,
                            gate_weights=None,
                            lambda_gate=0
                        )
                        loss += loss_i
                    loss /= nb_chunks

                raw_loss = loss.detach().clone()
                #mean_val_loss += raw_loss
                mean_val_loss += loss.item()

            mean_val_loss = mean_val_loss / self.len_val_epoch

        if nb_classes == 1:
            return mean_val_loss, NEW_outs, NEW_trgs 
        else:
            return mean_val_loss, NEW_outs[1:,:], NEW_trgs[1:]


    def _progress(self, batch_idx, train_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(train_loader, 'n_samples'):
            current = batch_idx * train_loader.batch_size
            total = train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model


    def _rebuild_optimizer_for_unfreeze(self):
        m = self._unwrap_model()

        gate_params = list(m.gate.parameters()) if hasattr(m, "gate") else []
        clf_params  = list(m.classifier.parameters()) if hasattr(m, "classifier") else []

        expert_params = []
        if hasattr(m, "experts"):
            for e in m.experts:
                expert_params += [p for p in e.parameters() if p.requires_grad]

        if len(expert_params) == 0:
            self.logger.metric("[Unfreeze] No expert params found/trainable. Skipping optimizer rebuild.")
            return

        base_group = self.optimizer.param_groups[0]
        base_lr = base_group.get("lr", 1e-3)

        opt_class = self.optimizer.__class__
        opt_defaults = getattr(self.optimizer, "defaults", {})

        head_params = [p for p in (gate_params + clf_params) if p.requires_grad]

        param_groups = [
            {"params": head_params, "lr": base_lr},
            {"params": expert_params, "lr": float(self.lr_experts)},
        ]

        new_kwargs = dict(opt_defaults)
        new_kwargs.pop("params", None)

        self.optimizer = opt_class(param_groups, **new_kwargs)

        try:
            if self.scheduler is not None:
                sched_class = self.scheduler.__class__
                if hasattr(self.scheduler, "state_dict"):
                    old_state = self.scheduler.state_dict()
                else:
                    old_state = None

                if sched_class.__name__ == "ReduceLROnPlateau":
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode=self.scheduler.mode,
                        factor=self.scheduler.factor,
                        patience=self.scheduler.patience,
                        threshold=self.scheduler.threshold,
                        threshold_mode=self.scheduler.threshold_mode,
                        cooldown=self.scheduler.cooldown,
                        min_lr=self.scheduler.min_lrs[0] if isinstance(self.scheduler.min_lrs, list) else self.scheduler.min_lrs,
                        eps=self.scheduler.eps,
                        verbose=getattr(self.scheduler, "verbose", False),
                    )
                else:
                    # fallback: keep scheduler as-is
                    pass

                if old_state is not None:
                    try:
                        self.scheduler.load_state_dict(old_state)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.metric(f"[Unfreeze] Scheduler rebuild failed: {e}")

    def train(self, nb_classes):
        """
        Full training logic
        """

        not_improved_count = 0
        final_val_outs, final_val_trgs, final_test_outs, final_test_trgs = [],[],[],[]
        
        if self.testmode == "NO":
            train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        drop_last=False,
                                                        num_workers=0)
            val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        num_workers=0)  
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       num_workers=0)    

        for epoch in range(self.start_epoch, self.epochs + 1):
            
            self.logger.metric("***** EPOCH : {} ********".format(epoch))

            if (self.testmode == "NO") and (epoch == int(self.unfreeze_epoch)):
                m = self._unwrap_model()

                if hasattr(m, "unfreeze_experts"):
                    self.logger.metric(f"[Unfreeze] Unfreezing experts at epoch {epoch}")
                    m.unfreeze_experts()

                    self.optimizer.param_groups[1]["lr"] = float(self.lr_experts)

                    self.logger.metric(f"[Unfreeze] Expert LR set to {self.lr_experts}")
                    
                    nb_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                    self.logger.metric(f"[Unfreeze] NUMBER OF PARAMETERS with Experts = {nb_params} \n")
                else:
                    self.logger.metric("[Unfreeze] Model has no unfreeze_experts() method.")


            if self.testmode == "NO":
                mean_train_loss, mean_val_loss, mean_test_loss, NEW_train_outs, NEW_train_trgs, NEW_val_outs, NEW_val_trgs, NEW_test_outs, NEW_test_trgs = self._train_epoch(train_loader, val_loader, test_loader, nb_classes)
            else:
                mean_train_loss, mean_val_loss, mean_test_loss, NEW_test_outs, NEW_test_trgs, NEW_molecules, NEW_doses, NEW_file_names = self._valid_epoch(test_loader, nb_classes)

            log = {'epoch': epoch, 'mean_train_loss': mean_train_loss, 'mean_val_loss': mean_val_loss, 'mean_test_loss': mean_test_loss}


            for key, value in log.items():
                if "loss" in str(key):
                    self.logger.metric("{}      : {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if epoch % self.save_period == 0 and self.testmode == "NO":
                self._save_checkpoint(epoch, save_best=best)


            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if self.testmode == "NO":
                collection_outs = [NEW_train_outs, NEW_train_trgs, NEW_val_outs, NEW_val_trgs, NEW_test_outs, NEW_test_trgs]
                out_text = ["Train", "", "Val", "", "Test", ""]
            else:
                collection_outs = [NEW_test_outs, NEW_test_trgs, NEW_molecules, NEW_doses, NEW_file_names]
                out_text = ["Test", ""]
                df_test_results = pd.DataFrame(NEW_test_outs.cpu().numpy(), columns=[f"Class_{i}" for i in range(nb_classes)])
                df_test_results['Target'] = NEW_test_trgs.cpu().numpy()
                df_test_results['Molecule'] = NEW_molecules
                df_test_results['Dose'] = NEW_doses
                df_test_results['File_Name'] = NEW_file_names
                df_test_results.to_csv(os.path.join(self.checkpoint_dir, self.output_file_name), index=False)
            for i in range(0,len(out_text),2):
                prefix_text = out_text[i]
                
                if nb_classes > 1:
                    outs = collection_outs[i][1:,:]
                    trgs = collection_outs[i+1][1:]#torch.from_numpy(trgs)

                    self.logger.metric(f"-------- {prefix_text} -----------")

                    metric_object = Metrics_Computation(outs,trgs,nb_classes)
                    bal_accuracy = metric_object.balanced_accuracy()
                    accuracy = metric_object.accuracy()
                    f1_score = metric_object.f1_score_macro()
                    precision = metric_object.precision()
                    recall = metric_object.recall()
                    auroc = metric_object.auroc()
                    auprc = metric_object.auprc()

                    self.logger.metric("{} Balanced ACC    : {}".format(prefix_text, bal_accuracy))
                    self.logger.metric("{} ACC    : {}".format(prefix_text, accuracy))
                    self.logger.metric("{} Precision     : {}".format(prefix_text, precision)) 
                    self.logger.metric("{} Recall     : {}".format(prefix_text, recall))  
                    self.logger.metric("{} F1_score     : {}".format(prefix_text, f1_score)) 
                    self.logger.metric("{} AUROC     : {}".format(prefix_text, auroc)) 
                    self.logger.metric("{} AUPRC     : {}".format(prefix_text, auprc))
                    if  self.testmode == "YES": 
                        self.logger.metric("{},{},{},{},{},{},{}".format(bal_accuracy, accuracy, precision, recall, f1_score, auroc, auprc)) 

                else:
                    outs = collection_outs[i]     # preds tensor [N]
                    trgs = collection_outs[i+1]   # targets tensor [N]
                    #print(f"outs : {outs.shape}")
                    #print(f"trgs : {trgs.shape}")
                    #print(np.unique(trgs))

                    self.logger.metric(f"-------- {prefix_text} -----------")

                    metric_object = RegressionMetrics(preds=outs, truths=trgs)
                    corr = metric_object.corrcoef()
                    r2   = metric_object.r2()
                    rmse = metric_object.rmse()

                    self.logger.metric(f"{prefix_text} Corrcoef : {corr}")
                    self.logger.metric(f"{prefix_text} R2       : {r2}")
                    self.logger.metric(f"{prefix_text} RMSE     : {rmse}")
                    if  self.testmode == "YES": 
                        self.logger.metric("{},{},{},{},{},{},{}".format(corr, r2, rmse))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            if nb_classes > 1:
                if f1_score > self.best_val_f1_score and self.testmode == "NO":
                    self.best_val_f1_score = f1_score
                    self._save_checkpoint(epoch, save_best=True)
            else:
                if rmse > self.best_val_rmse and self.testmode == "NO":
                    self.best_val_rmse = rmse
                    self._save_checkpoint(epoch, save_best=True)

    def _prepare_device(self, n_gpu_use):

        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best == False:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        else:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")


    def _resume_checkpoint(self, resume_path):

        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))






