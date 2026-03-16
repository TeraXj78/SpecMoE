import os
import torch
import numpy as np
from torch.nn import MSELoss
from torchinfo import summary
from ptflops import get_model_complexity_info
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from Utils.loss import multiple_loss_function
from Utils.Positional_Encoding import RotaryPE             

import wandb
import matplotlib.pyplot as plt
import re


def plot_grad_flow_grouped(named_parameters, epoch):

    named_parameters = list(named_parameters)

    combo_groups = [
        ("Down1", "features1"), ("Down1", "features3"),
        ("Down2", "features1"), ("Down2", "features3"),
        ("Down3", "features1"), ("Down3", "features3"),
        ("Up1", "features1"), ("Up1", "features3"),
        ("Up2", "features1"), ("Up2", "features3"),
        ("Up3", "features1"), ("Up3", "features3"),
    ]
    single_groups = [
        "Bottom1", "Bottom2",
        "TransformerEnc1", "TransformerEnc2", "TransformerEnc3",
        "TimeSet_Pooling_1", "TimeSet_Pooling_2"
    ]


    grad_stats = {}
    for g1, g2 in combo_groups:
        grad_stats[f"{g1}_{g2}"] = {"mean": [], "max": []}
    for g in single_groups:
        grad_stats[g] = {"mean": [], "max": []}
    grad_stats["Other"] = {"mean": [], "max": [], "names": []}

    for name, param in named_parameters:
        if not (param.requires_grad and param.grad is not None):
            continue

        matched = False

        # Combo groups
        for g1, g2 in combo_groups:
            if g1 in name and g2 in name:
                grad_stats[f"{g1}_{g2}"]["mean"].append(param.grad.abs().mean().item())
                grad_stats[f"{g1}_{g2}"]["max"].append(param.grad.abs().max().item())
                matched = True
                break

        # Single groups
        if not matched:
            for g in single_groups:
                if g in name:
                    grad_stats[g]["mean"].append(param.grad.abs().mean().item())
                    grad_stats[g]["max"].append(param.grad.abs().max().item())
                    matched = True
                    break

        # Everything else
        if not matched:
            grad_stats["Other"]["mean"].append(param.grad.abs().mean().item())
            grad_stats["Other"]["max"].append(param.grad.abs().max().item())
            grad_stats["Other"]["names"].append(name)


    other_names = grad_stats["Other"]["names"]
    other_groups = {}

    for name in other_names:
        prefix = name.split('.')[0]
        prefix = re.sub(r"\d+$", "", prefix)
        key = f"Other_{prefix}"
        if key not in other_groups:
            other_groups[key] = {"mean": [], "max": []}

        param = next((p for n, p in named_parameters if n == name), None)
        if param is None or param.grad is None:
            continue

        other_groups[key]["mean"].append(param.grad.abs().mean().item())
        other_groups[key]["max"].append(param.grad.abs().max().item())

    for key, vals in other_groups.items():
        if len(vals["mean"]) >= 3:
            grad_stats[key] = vals

    grad_stats.pop("Other")


    groups, ave_means, max_means = [], [], []
    for key, stats in grad_stats.items():
        if stats["mean"]:
            groups.append(key)
            ave_means.append(sum(stats["mean"]) / len(stats["mean"]))
            max_means.append(sum(stats["max"]) / len(stats["max"]))

    if not groups:
        print(f"[Warning] No gradients found at epoch {epoch}")
        return

    plt.figure(figsize=(max(14, len(groups) * 0.5), 6))
    x = range(len(groups))
    plt.bar(x, max_means, alpha=0.4, color='c', label='max grad')
    plt.bar(x, ave_means, alpha=0.4, color='b', label='mean grad')
    plt.xticks(x, groups, rotation=70, ha='right', fontsize=8)
    plt.ylabel("Gradient Magnitude")
    plt.title(f"Grouped Gradient Flow (epoch {epoch})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.log({f"grad_flow_grouped": wandb.Image(plt)})
    plt.close()


    wandb.log({
        **{f"grad_group_mean/{g}": ave_means[i] for i, g in enumerate(groups)},
        **{f"grad_group_max/{g}": max_means[i] for i, g in enumerate(groups)}
    })





# ------------------------------------------------------------------------------------------
def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: No GPU available, switching to CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: requested {n_gpu_use} GPUs, but only {n_gpu} available.")
        n_gpu_use = n_gpu
    device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
# ------------------------------------------------------------------------------------------

class Trainer(object):
    def __init__(self, config, params, data_loader, model):

        self.checkpoint_dir = config.save_dir
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = self.checkpoint_dir / "runs" / timestamp
        self.writer = SummaryWriter(log_path)

        self.params = params
        self.device, device_ids = prepare_device(self.params.n_gpus)
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.PE_object = RotaryPE(self.device)

        self.criterion = multiple_loss_function(aux_weights=(0.3,0.2), 
                                                term_weights={
                                                    "mse": 1.0,
                                                    "spec": 0.02,}
                                                )
                                               

        self.wandb_status = params.wandb_status
        if self.wandb_status == "ON":
            base_dir = "..../wandb_logs"
        
            iddd =  f"{self.params.Experiment_name}" 
        
            #os.makedirs(exp_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = base_dir
        
            wandb.init(
                project=self.params.project_name,
                name=self.params.Experiment_name,
                config=vars(self.params),
                mode="offline",
                id=iddd,
            )
            wandb.watch(self.model)
        # Multi-GPU
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


        # Optimizer & LR scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay
        )

        if self.params.lr_scheduler == 'CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.params.epochs * len(self.data_loader), eta_min=1e-6)
        elif self.params.lr_scheduler == 'ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999)
        elif self.params.lr_scheduler == 'StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5 * len(self.data_loader), gamma=0.5)
        elif self.params.lr_scheduler == 'MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10, 20, 30], gamma=0.1)
        elif self.params.lr_scheduler == 'CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001,
                step_size_up=len(self.data_loader) * 5,
                step_size_down=len(self.data_loader) * 2,
                mode='exp_range', gamma=0.9, cycle_momentum=False)
        else:
            self.optimizer_scheduler = None

        # Load Optimizer & scheduler if we are one resume mode
        if self.params.resume == "NO":
            self.start_epoch = 0
        elif self.params.resume == "YES":
            self.optimizer.load_state_dict(self.params.optimizer_state)
            self.optimizer_scheduler.load_state_dict(self.params.scheduler_state)
            self.start_epoch         = self.params.last_epoch + 1
            

    # ----- Save checkpoint -----
    def save_checkpoint(self, epoch, best_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.optimizer_scheduler.state_dict() if self.optimizer_scheduler is not None else None,
            "best_loss": best_loss,
            "params": vars(self.params),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        if (epoch+1)%10 == 0:
            ckpt_path_10 = self.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
            torch.save(checkpoint, ckpt_path_10)
            print(f"Checkpoint saved: {ckpt_path_10}")
    
        if is_best:
            ckpt_path = self.checkpoint_dir / f"best_model.pth"
        else:
            ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
    
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    # --------------------------------------------------------------------------------------


    
    def train(self):
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(self.start_epoch, self.params.epochs):
            self.model.train()
            epoch_losses, epoch_mse_losses, epoch_spec_losses, epoch_aux2_losses, epoch_aux1_losses = [],[],[],[],[]

            self.optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{self.params.epochs}", mininterval=10)
            for step_i, (amplit_arrays, times_array) in enumerate(pbar):

                # Amplitudes normalized to stay between -1 and 1 
                amplit_arrays = amplit_arrays / 100

                amplit_arrays = amplit_arrays.to(self.device)   
                
                pos_enc = times_array.to(self.device)      
                pos_enc = pos_enc.unsqueeze(1)
                """
                print(f"step_i        : {step_i}")
                print(f"amplit_arrays : {amplit_arrays.shape}")
                print(f"pos_enc       : {pos_enc.shape}")
                """
                
    
                out_Feats1, out_Feats3, aux_u2, aux_u1_attn = self.model(amplit_arrays, pos_enc, self.PE_object) 
                targets = amplit_arrays

                
                # Compute masked reconstruction loss
                loss, mse_loss, spectral_loss, loss_aux2, loss_aux1 = self.criterion([out_Feats1, out_Feats3], targets, aux_outputs=[aux_u2,aux_u1_attn])
                    

                # Backward
                (loss / self.params.Accumulation_Grad).backward()
                
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                if (step_i+1) % self.params.Accumulation_Grad == 0 or (step_i+1) == len(self.data_loader):
                    self.optimizer.step()
                    if self.optimizer_scheduler is not None:
                        self.optimizer_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_losses.append(loss.item())
                epoch_mse_losses.append(mse_loss.item())
                epoch_spec_losses.append(spectral_loss.item())
                epoch_aux2_losses.append(loss_aux2.item())
                epoch_aux1_losses.append(loss_aux1.item())
                global_step += 1

                pbar.set_postfix({"loss": np.mean(epoch_losses)})

            # Logging
            mean_loss = np.mean(epoch_losses)
            mean_mse_loss = np.mean(epoch_mse_losses)
            mean_spec_loss = np.mean(epoch_spec_losses)
            mean_aux2_loss = np.mean(epoch_aux2_losses)
            mean_aux1_loss = np.mean(epoch_aux1_losses)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Total_Loss: {mean_loss:.6f} | mse_Loss: {mean_mse_loss:.6f} | spec_Loss: {mean_spec_loss:.6f}  | aux2_loss: {mean_aux2_loss:.6f} | aux1_loss: {mean_aux1_loss:.6f} | LR: {lr:.2e}")

            self.writer.add_scalar("Loss/train", mean_loss, epoch)
            self.writer.add_scalar("LR", lr, epoch)

            # Log gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)
                    self.writer.add_histogram(f"weights/{name}", param, epoch)

            # --- global gradient stats ---
            total_grad_sum, total_grad_count = 0.0, 0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_grad_sum += p.grad.abs().sum().item()
                    total_grad_count += p.grad.numel()

            grad_mean = total_grad_sum / total_grad_count if total_grad_count > 0 else 0.0

            if self.wandb_status == "ON":
                plot_grad_flow_grouped(self.model.named_parameters(), epoch)
                wandb.log({
                    "epoch": epoch + 1,
                    "total_loss": mean_loss,
                    "mse_loss": mean_mse_loss,
                    "spectral_loss": mean_spec_loss,
                    "aux2_loss": mean_aux2_loss,
                    "aux1_loss": mean_aux1_loss,
                    "learning_rate": lr,
                    "grad_sum": total_grad_sum,
                    "grad_mean": grad_mean,
                })            

            # Save model
            if mean_loss < best_loss:
                best_loss = mean_loss
                self.save_checkpoint(epoch, best_loss, is_best=True)
            elif (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, best_loss, is_best=False)
        
        if self.wandb_status == "ON":
            wandb.finish()













