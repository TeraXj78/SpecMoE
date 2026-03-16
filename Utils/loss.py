import torch
import torch.nn as nn
import torch.nn.functional as F



class multiple_loss_function(nn.Module):
    def __init__(self, aux_weights=(0.3,0.2), term_weights=None):
        super().__init__()
        self.aux_weights = aux_weights
        self.term_weights = term_weights or {
            "mse": 0.6,
            "spec": 0.4,
        }

    # ---------- LOSS COMPONENTS ----------
    def spectral_loss(self, pred, target, n_fft=512, hop_length=256, win_length=512):
        """
        Spectral loss between predicted and target signals, applied channel-wise.
        """
        B, C, T = pred.shape
        device = pred.device
    
        # STFT per channel
        loss = 0.0
        for c in range(C):
            pred_fft = torch.stft(
                pred[:, c, :], n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=torch.hann_window(win_length, device=device), return_complex=True, 
                center=True, normalized=False
            )
            target_fft = torch.stft(
                target[:, c, :], n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=torch.hann_window(win_length, device=device), return_complex=True,
                center=True, normalized=False
            )
            loss += torch.mean((torch.abs(pred_fft) - torch.abs(target_fft)) ** 2)
        return loss / C

    def grad_diff_loss(self, pred, target):
        grad_pred = pred[..., 1:] - pred[..., :-1]
        grad_targ = target[..., 1:] - target[..., :-1]
        return F.l1_loss(grad_pred, grad_targ)

    def corr_loss(self, pred, target):
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        return (1 - cos_sim).mean()

    def base_loss(self, pred, target):
        """composite loss on one output"""
        mse = F.mse_loss(pred, target)
        spec = self.spectral_loss(pred, target)

        tw = self.term_weights

        return (tw["mse"] * mse +
                tw["spec"] * spec), tw["mse"]*mse, tw["spec"]*spec
        
    # ---------- TOTAL LOSS ----------
    def forward(self, x0_pred, x0_true, aux_outputs=None):

        # Main output loss
        loss_main1, mse1, spec1 = self.base_loss(x0_pred[0], x0_true)
        loss_main2, mse2, spec2 = self.base_loss(x0_pred[1], x0_true)
        loss_main, mse, spec = 1*(loss_main1+loss_main2), 1*(mse1+mse2), 1*(spec1+spec2)

        # Auxiliary outputs
        loss_aux = 0.0
        if aux_outputs is not None:
            i = 0
            for aux_pred, w in zip(aux_outputs, self.aux_weights):
                target_down = F.interpolate(x0_true, size=aux_pred.shape[-1], mode='linear', align_corners=False)
                loss_aux_unWeighted, _, _ = self.base_loss(aux_pred, target_down)
                loss_i = w * loss_aux_unWeighted
                loss_aux += loss_i
                if i == 0:
                    loss_aux2 = loss_i
                    i+=1
                if i == 1:
                    loss_aux1 = loss_i

            total_loss = loss_main + loss_aux
            return total_loss, mse, spec, loss_aux2, loss_aux1
            
        else:
            total_loss = loss_main
            return total_loss, mse, spec






def regression_Loss(
    output,
    target,
    classes_weights,
    device,
    gate_weights=None,          # w: [B, 3]
    lambda_gate=0.01,           # regularization strength
    eps=1e-8
):
    # ----- Regression loss -----
    criterion = torch.nn.MSELoss()
    ce_loss = criterion(output, target)

    # ----- Gate entropy regularization -----
    if gate_weights is not None:
        # entropy per sample
        entropy = -torch.sum(
            gate_weights * torch.log(gate_weights + eps),
            dim=1
        ).mean()

        total_loss = ce_loss - (lambda_gate * entropy)
        return total_loss, entropy.detach()

    return ce_loss




    
def weighted_CrossEntropyLoss(
    output,
    target,
    classes_weights,
    device,
    gate_weights=None,          
    lambda_gate=0.01,           
    eps=1e-8
):

    # ----- Classification loss -----
    criterion = nn.CrossEntropyLoss(weight=classes_weights)
    ce_loss = criterion(output, target)

    # ----- Gate entropy regularization -----
    if gate_weights is not None:
        entropy = -torch.sum(
            gate_weights * torch.log(gate_weights + eps),
            dim=1
        ).mean()

        total_loss = ce_loss + (lambda_gate * entropy)
        return total_loss, entropy.detach()

    return ce_loss






















    