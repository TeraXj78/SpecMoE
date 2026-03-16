
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
from copy import deepcopy
from einops import rearrange, repeat

torch.autograd.set_detect_anomaly(True)





    
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
    
class TimeSet_Pooling_1(nn.Module):
    def __init__(self):
        super(TimeSet_Pooling_1, self).__init__()
        self.features1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=8, stride=5, padding=4),
            nn.AvgPool1d(kernel_size=16, stride=6, padding=6)
        )

    def forward(self, x, nb_channels):
        x1 = self.features1(x)
        x_concat = repeat(x1, "... L -> ... (L r)", r=nb_channels)
        return x_concat

class TimeSet_Pooling_2(nn.Module):
    def __init__(self):
        super(TimeSet_Pooling_2, self).__init__()
        self.features1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=8, stride=5, padding=4),
            nn.AvgPool1d(kernel_size=32, stride=12, padding=12)
        )

    def forward(self, x, nb_channels):
        x1 = self.features1(x)
        x_concat = repeat(x1, "... L -> ... (L r)", r=nb_channels)
        return x_concat


class FILM(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, out_dim)
        self.beta  = nn.Linear(emb_dim, out_dim)

    def forward(self, x, emb):
        # x: [B, C, T] 
        # emb: [B, emb_dim]
        gamma = self.gamma(emb).unsqueeze(-1) # [B, feat_dim, 1]
        beta  = self.beta(emb).unsqueeze(-1)  # [B, feat_dim, 1]

        return x * (1 + gamma) + beta
        


class Down1(nn.Module):
    def __init__(self, in_dim=1, mid_dim=32, out_dim=64, drate = 0.5):
        super(Down1, self).__init__()

        self.GELU = nn.GELU()  

        self.features1 = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, kernel_size=4, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(mid_dim),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=5, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(mid_dim, out_dim, kernel_size=4, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.Conv1d(out_dim, out_dim, kernel_size=4, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.MaxPool1d(kernel_size=16, stride=3, padding=7)
        )


        self.features3 = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, kernel_size=65, stride=3, dilation=4, bias=False, padding=97),
            nn.BatchNorm1d(mid_dim),
            self.GELU,
            nn.MaxPool1d(kernel_size=18, stride=5, padding=9),
            nn.Dropout(drate),

            nn.Conv1d(mid_dim, out_dim, kernel_size=16, stride=1, bias=False, padding=8),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.Conv1d(out_dim,out_dim, kernel_size=16, stride=1, bias=False, padding=8),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.MaxPool1d(kernel_size=16, stride=1, padding=8)
        )

        self.dropout = nn.Dropout(drate)

        #self.film = FILM(256,64)

    def forward(self, x, emb=None):
        B, one, C, T = x.shape
        x = x.view(B, 1, C*T)

        x1 = self.features1(x) # [B, D, C*T']
        x3 = self.features3(x) # [B, D, C*T']
        #print(f"!!! D1 x1 {x1.shape[-1]/19} - x3 {x3.shape[-1]/19} !!! ")
        
        x1 = self.dropout(x1)
        x3 = self.dropout(x3)

        return [x1, x3]



class Down2(nn.Module):
    def __init__(self, in_dim=64, mid_dim=64, out_dim=128, drate = 0.5):
        super(Down2, self).__init__()

        self.GELU = nn.GELU()  

        self.features1 = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, kernel_size=4, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(mid_dim),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(mid_dim, out_dim, kernel_size=4, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.Conv1d(out_dim, out_dim, kernel_size=4, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.MaxPool1d(kernel_size=8, stride=2, padding=3)
        )

        self.features3 = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, kernel_size=16, stride=1, dilation=2, bias=False, padding=11),
            nn.BatchNorm1d(mid_dim),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(mid_dim, out_dim, kernel_size=16, stride=1, bias=False, padding=8),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.Conv1d(out_dim,out_dim, kernel_size=16, stride=1, bias=False, padding=8),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.MaxPool1d(kernel_size=8, stride=2, padding=4)
        )

        self.dropout = nn.Dropout(drate)

        #self.film = FILM(256,128)


    def forward(self, list_x, C, emb=None):
        #seqLen = x.size(-1)
        #x_a, x_b = x[:,:,0:seqLen//2], x[:,:,seqLen//2:]
        x1 = self.features1(list_x[0])   # [B, D, C*T']
        x3 = self.features3(list_x[1])   # [B, D, C*T']
        #print(f"!!! D2 x1 {x1.shape[-1]/19} - x3 {x3.shape[-1]/19} !!! ")

        B, D, _ = x1.shape
        x1 = x1.view(B, D, C, -1)
        x3 = x3.view(B, D, C, -1)
        x_concat = torch.cat((x1, x3), dim=3)   # [B, D, C, 2T']
        x_concat = self.dropout(x_concat)

        return x_concat



class Down3(nn.Module):
    def __init__(self, in_dim=128, mid_dim=128, out_dim=128, drate = 0.5):
        super(Down3, self).__init__()

        self.GELU = nn.GELU()  

        self.features1 = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, kernel_size=4, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(mid_dim),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            nn.Dropout(drate),

            nn.Conv1d(mid_dim, out_dim, kernel_size=4, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,

        )

        self.features3 = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, kernel_size=8, stride=1, dilation=2, bias=False, padding=5),
            nn.BatchNorm1d(mid_dim),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(mid_dim, out_dim, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(out_dim),
            self.GELU,

        )

        self.dropout = nn.Dropout(drate)



    def forward(self, x, emb=None):
        B, D, C, T_dim = x.shape
        half_T_dim = T_dim//2
        x1, x3 = x[:,:,:,0:half_T_dim], x[:,:,:,half_T_dim:]
        x1, x3 = x1.contiguous().view(B, D, C*half_T_dim), x3.contiguous().view(B, D, C*half_T_dim)

        x1 = self.features1(x1)   # [B, D, C*half_T_dim_prim]
        x3 = self.features3(x3)   # [B, D, C*half_T_dim_prim]
        B, D, _ = x1.shape
        #print(f"!!! D3 x1 {x1.shape[-1]/19} - x3 {x3.shape[-1]/19} !!! ")

        x1 = x1.contiguous().view(B, D, C, -1)
        x3 = x3.contiguous().view(B, D, C, -1)
        
        #print(f"D3 x1{x1.shape} - x3{x3.shape}")
        x_concat = torch.cat((x1, x3), dim=3)  # [B, D, C, 2*half_T_dim_prim]
        x_concat = self.dropout(x_concat)
            
        return x_concat




class Bottom1(nn.Module):
    def __init__(self, in_dim=128, out_dim=256, drate = 0.5):
        super(Bottom1, self).__init__()

        self.GELU = nn.GELU()  

        self.features = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=8, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,
            nn.Dropout(drate),

            nn.Conv1d(out_dim, out_dim, kernel_size=8, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,

        )

    def forward(self, x, emb=None):
        
        x = self.features(x)

        return x

        
class Bottom2(nn.Module):
    def __init__(self, in_dim=256, out_dim=128, drate = 0.5):
        super(Bottom2, self).__init__()

        self.GELU = nn.GELU()  

        self.features = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=8, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,
            nn.Dropout(drate),

            nn.Conv1d(out_dim, out_dim, kernel_size=8, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,

        )

    def forward(self, x, emb=None):
        
        x = self.features(x)

        return x
##########################################################################################







##########################################################################################
class Up1(nn.Module):
    def __init__(self, in_dim=128, skip_dim=128, mid_dim=128, out_dim=128, drate=0.5):
        super().__init__()
        self.GELU = nn.GELU()

        in_cat = in_dim + skip_dim

        # features1 branch (mirror of Down3.features1)
        self.features1 = nn.Sequential(
            nn.Conv1d(in_cat, mid_dim, kernel_size=4, padding="same"),
            nn.BatchNorm1d(mid_dim),
            self.GELU,

            nn.ConvTranspose1d(mid_dim, mid_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(mid_dim),
            self.GELU,

            nn.Conv1d(mid_dim, out_dim, kernel_size=4, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,
        )


        # features3 branch (mirror of Down3.features3)
        self.features3 = nn.Sequential(
            nn.Conv1d(in_cat, mid_dim, kernel_size=8, padding=4),
            nn.BatchNorm1d(mid_dim),
            self.GELU,

            nn.ConvTranspose1d(mid_dim, mid_dim, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(mid_dim),
            self.GELU,

            nn.Dropout(drate),

            nn.Conv1d(mid_dim, out_dim, kernel_size=8, dilation=2, padding=7),
            nn.BatchNorm1d(out_dim),
            self.GELU,
        )


    def forward(self, x, skip, emb=None):
        B, D, C, T3 = x.shape
        skip = skip.contiguous().view(B, D, C, T3)
        x = torch.cat([x, skip], dim=1)  # [B, 2D, C, T3]
        #print(f" cat([x, skip]  {x.shape}")

        # split along sequence dim into two halves (features1 / features3)
        x1, x3 = x[:, :, :, :T3//2], x[:, :, :, T3//2:]
        x1 = x1.contiguous().view(B, 2*D, -1)  # [B, 2D, C*T3//2]
        x3 = x3.contiguous().view(B, 2*D, -1)  # [B, 2D, C*T3//2]
        
        # branch processing at same resolution
        y1 = self.features1(x1)  # [B, D, C*T4]
        y3 = self.features3(x3)  # [B, D, C*T4]
        B, D, _ = y1.shape
        #print(f"!!! U1 y1 {y1.shape[-1]/19} - y3 {y3.shape[-1]/19} !!! ")

        y1 = y1.contiguous().view(B, D, C, -1)
        y3 = y3.contiguous().view(B, D, C, -1)

        
        y_concat = torch.cat([y1, y3], dim=3)


        return y_concat



class Up2(nn.Module):
    def __init__(self, in_dim=128, skip_dim=128,  mid_dim=64, out_dim=64, drate=0.5):
        super().__init__()
        self.GELU = nn.GELU()
        in_cat = in_dim + skip_dim

        self.features1 = nn.Sequential(
            nn.ConvTranspose1d(in_cat, in_dim, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(in_dim),
            self.GELU,

            nn.Conv1d(in_dim, in_dim, kernel_size=4, padding="same"),
            nn.BatchNorm1d(in_dim),
            self.GELU,
            nn.Conv1d(in_dim, out_dim, kernel_size=4, padding="same"),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.Dropout(drate),

            nn.ConvTranspose1d(out_dim, out_dim, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.Conv1d(out_dim, out_dim, kernel_size=4, padding="same")
        )

        # features3: mirror of Down2.features3
        self.features3 = nn.Sequential(
            nn.ConvTranspose1d(in_cat, in_dim, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(in_dim),
            self.GELU,

            nn.Conv1d(in_dim, in_dim, kernel_size=16, padding=8),
            nn.BatchNorm1d(in_dim),
            self.GELU,
            nn.Conv1d(in_dim, out_dim, kernel_size=16, padding=8),
            nn.BatchNorm1d(out_dim),
            self.GELU,

            nn.Dropout(drate),

            nn.ConvTranspose1d(out_dim, out_dim, kernel_size=8, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(out_dim),
            self.GELU,
            nn.ConvTranspose1d(out_dim, out_dim, kernel_size=16, stride=2, dilation=2, padding=16, output_padding=1) #11
        )


    def forward(self, x, skip, emb=None):
        B, D, C, T4 = x.shape     # T4 = T2
        B, D, C, T2 = skip.shape  # T4 = T2

        x = torch.cat([x, skip], dim=1)  # [B, 2D, C, T4]
        
        # split along sequence dim into two halves (features1 / features3)
        x1, x3 = x[:, :, :, :T4//2], x[:, :, :, T4//2:]
        x1 = x1.contiguous().view(B, 2*D, -1)  # [B, 2D, C*T4//2]
        x3 = x3.contiguous().view(B, 2*D, -1)  # [B, 2D, C*T4//2]
        #print(f"x1  {x1.shape} ")
        #print(f"x3  {x3.shape} ")
        
        # branch processing at same resolution
        y1 = self.features1(x1)  # [B, D, C*T5]
        y3 = self.features3(x3)  # [B, D, C*T5]
        B, D, _ = y1.shape 
        #print(f"!!! U2 y1 {y1.shape[-1]/19} - y3 {y3.shape[-1]/19} !!! ")

        y1 = y1.contiguous().view(B, D, C, -1)
        y3 = y3.contiguous().view(B, D, C, -1)

        y_concat = torch.cat([y1, y3], dim=3)


        return y_concat



class Up3(nn.Module):
    def __init__(self, in_dim=64, skip_dim=64, mid_dim=32, out_dim=1, drate=0.5):
        super().__init__()
        self.GELU = nn.GELU()
        in_cat = in_dim + skip_dim

        self.features1 = nn.Sequential(
            nn.ConvTranspose1d(in_cat, in_dim, kernel_size=16, stride=5, padding=7),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.Conv1d(in_dim, in_dim, kernel_size=4, padding=2),
            nn.BatchNorm1d(in_dim),
            self.GELU,
            nn.Conv1d(in_dim, mid_dim, kernel_size=4, padding=2),
            nn.BatchNorm1d(mid_dim),
            self.GELU,

            nn.Dropout(drate),

            nn.ConvTranspose1d(mid_dim, mid_dim, kernel_size=8, stride=3, padding=1, output_padding=1),
            nn.BatchNorm1d(mid_dim),
            self.GELU,

            nn.Conv1d(mid_dim, out_dim, kernel_size=4, padding=1)
        )

        # features3 mirrors Down1.features3
        self.features3 = nn.Sequential(
            nn.ConvTranspose1d(in_cat, in_dim, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(in_dim),
            self.GELU,

            nn.Conv1d(in_dim, in_dim, kernel_size=16, padding=8),
            nn.BatchNorm1d(in_dim),
            self.GELU,
            nn.Conv1d(in_dim, mid_dim, kernel_size=16, padding=8),
            nn.BatchNorm1d(mid_dim),
            self.GELU,

            nn.Dropout(drate),

            nn.ConvTranspose1d(mid_dim, mid_dim, kernel_size=18, stride=5, padding=9),
            nn.BatchNorm1d(mid_dim),
            self.GELU,

            nn.ConvTranspose1d(mid_dim, out_dim, kernel_size=65, stride=3, dilation=4, padding=127, output_padding=0)
        )

       

    def forward(self, x, list_skip, emb=None):
        B, D, C, T5 = x.shape
        skip = torch.cat(list_skip, axis=2)
        skip = skip.view(B, D, C, T5)

        x = torch.cat([x, skip], dim=1)  # [B, 2D, C, T5]
        
        # split along sequence dim into two halves (features1 / features3)
        x1, x3 = x[:, :, :, :T5//2], x[:, :, :, T5//2:]
        x1 = x1.contiguous().view(B, 2*D, -1)  # [B, 2D, C*T5//2]
        x3 = x3.contiguous().view(B, 2*D, -1)  # [B, 2D, C*T5//2]
        
        # branch processing at same resolution
        y1 = self.features1(x1)  # [B, D, C*T6]
        y3 = self.features3(x3)  # [B, D, C*T6]
        B, D, _ = y1.shape
        #print(f"!!! U3 y1 {y1.shape[-1]/19} - y3 {y3.shape[-1]/19} !!! ")
        
        #print(f"y1  {y1.shape[-1]/19} ")
        #print(f"y3  {y3.shape[-1]/19} ")

        y1 = y1.contiguous().view(B, D, C, -1)
        y3 = y3.contiguous().view(B, D, C, -1)


        return y1, y3


########################################################################################################################################





class MultiHeadedAttention(nn.Module):
    def __init__(self, nb_head, dim_emb, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert dim_emb % nb_head == 0
        self.head_dim = dim_emb // nb_head
        self.nb_head = nb_head
        #self.seqLen = d_model

        self.k_proj = nn.Linear(dim_emb, dim_emb)
        self.v_proj = nn.Linear(dim_emb, dim_emb)
        self.out_proj = nn.Linear(dim_emb, dim_emb)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, pos_enc, PE_object):
        "Implements Multi-head attention"
        batch_size, seq_length, dim_emb = query.size()

        query = query.reshape(batch_size, self.nb_head, seq_length, self.head_dim)
        key   = self.k_proj(key).reshape(batch_size, self.nb_head, seq_length, self.head_dim)
        value = self.v_proj(value).reshape(batch_size, self.nb_head, seq_length, self.head_dim)
        
        "RoPE : Rotary Positional Encoding; "
        #"frequency-encode positions"
        freq_pos_enc = PE_object.freq_pos_enc(pos_enc, dim=self.head_dim) #None
        #"Adding absolute positions into queries and keys"
        query_rot = PE_object.rotate(query, freq_pos_enc)
        key_rot   = PE_object.rotate(key, freq_pos_enc)
        value_rot   = PE_object.rotate(value, freq_pos_enc)

        "Attention_layer"
        scores = torch.matmul(query_rot, key_rot.transpose(-2, -1)) / (self.head_dim**0.5)

        #----------------------- Common SoftMax-------------------------------
        #attn_weights = F.softmax(scores, dim=-1)
        #---------------------------------------------------------------------
        #print(f"DIm scores {scores.shape}")
        #----------------------- Making Sure each row excluding the diagonal sums to 1
        diag1 = torch.tril(scores, diagonal=-1)
        diag2 = torch.triu(scores, diagonal=1)
        attn_weights = F.softmax(diag1+diag2, dim=2) #[B,H,L,L]
        del diag1, diag2, scores
        #---------------------------------------------------------------------      
        attn_output = torch.matmul(attn_weights, value_rot)

        x = attn_output.transpose(1, 2).reshape(batch_size, seq_length, dim_emb)
        return self.out_proj(x)
    


class LayerNorm(nn.Module):
    "Construct a layer normalization module."
    def __init__(self, dim_emb, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(dim_emb))
        self.b_2 = nn.Parameter(torch.zeros(dim_emb))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return (self.a_2 * x) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''
    def __init__(self, dim_emb, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(dim_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        z = sublayer(x)
        x = self.norm(x + z)
        return x 


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])




class Transformer_EncoderLayer(nn.Module):
    '''
    An encoder layr
    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, MultiH_Attn, feed_forward, dim_emb, dropout):
        super(Transformer_EncoderLayer, self).__init__()
        self.MultiH_Attn = MultiH_Attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(dim_emb, dropout), 2)
        self.lin_1 = nn.Linear(dim_emb, dim_emb)

    def forward(self, x_1, x_2, pos_enc, PE_object, emb=None):
        
        "Transformer Encoder"
        query = self.lin_1(x_1)
        x = self.sublayer_output[0](query, lambda x: self.MultiH_Attn(query, x_2, x_1, pos_enc, PE_object))
        out = self.sublayer_output[1](x, self.feed_forward)

        return out


class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network"
    def __init__(self, dim_emb, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_emb, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_emb)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))







class CoSup_UNet_SSL(nn.Module):
    def __init__(self, nb_attn_heads):
        super(CoSup_UNet_SSL, self).__init__()

        self.TimeSet_Pooling_1 = TimeSet_Pooling_1()
        self.TimeSet_Pooling_2 = TimeSet_Pooling_2()
        
        dp_proba = 0.2
        
        attn = MultiHeadedAttention(nb_attn_heads, 64)
        ff = PositionwiseFeedForward(64, 128, dropout=dp_proba)
        attn2 = MultiHeadedAttention(nb_attn_heads, 128)
        ff2 = PositionwiseFeedForward(128, 256, dropout=dp_proba)

        self.Down1 = Down1(in_dim=1, mid_dim=32, out_dim=64, drate=dp_proba)
        self.Down2 = Down2(in_dim=64, mid_dim=64, out_dim=64, drate=dp_proba)
        self.TransformerEnc1 = Transformer_EncoderLayer(deepcopy(attn), deepcopy(ff), 64, dropout=dp_proba)
        self.Down3 = Down3(in_dim=64, mid_dim=128, out_dim=128, drate=dp_proba)


        self.Bottom1 = Bottom1(in_dim=128, out_dim=128, drate=dp_proba)
        self.TransformerEnc2 = Transformer_EncoderLayer(attn2, ff2, 128, dropout=dp_proba)
        self.Bottom2 = Bottom2(in_dim=128, out_dim=128, drate=dp_proba)

        
        self.Up1 = Up1(in_dim=128, skip_dim=128, mid_dim=128, out_dim=64, drate=dp_proba)  
        self.TransformerEnc3 = Transformer_EncoderLayer(deepcopy(attn), deepcopy(ff), 64, dropout=dp_proba)
        self.Up2 = Up2(in_dim=64, skip_dim=64,  mid_dim=64, out_dim=64, drate=dp_proba)
        self.Up3 = Up3(in_dim=64, skip_dim=64, mid_dim=32, out_dim=1, drate=dp_proba)

        self.aux_head_up2 = nn.Sequential(
            nn.Linear(64, 19),   # maps embedding to EEG channels
        )
        self.aux_head_up1 = nn.Sequential(
            nn.Linear(64, 19),   # maps embedding to EEG channels
        )
        



    def forward(self, input_x, absolute_PE, PE_object, mask=None):

        #input_x_pref, input_x_par = input_x
        B, C, T = input_x.shape
        input_x = input_x.unsqueeze(1)
        #input_x = input_x.view(B, 1, C*T)
        

        list_D1 = self.Down1(input_x)
        #print(f"----> D1 Enc1 {list_D1[0].shape}")
        #print(f"----> D1 Enc2 {list_D1[1].shape}")
        D2 = self.Down2(list_D1, C)   # [B, D, C, 2T']
        #print(f"----> D2 {D2.shape}")
        B, D, C, T2 = D2.shape
        D2 = D2.view(B, C*T2, D)
        #print(f"----> D2 {D2.shape}")
        
        # attention between encoder stages
        pos_enc_1 = self.TimeSet_Pooling_1(absolute_PE, C)
        #print(f"----> pos_enc_1 {pos_enc_1.shape}")
        D2_attn = self.TransformerEnc1(D2, D2, pos_enc_1, PE_object)
        #print(f"D2_attn {D2_attn.shape}")
        D2_attn = D2_attn.permute(0,2,1).view(B, D, C, T2)
        #print(f"D2_attn {D2_attn.shape}")
        

        D3 = self.Down3(D2_attn)  # [B, D, C, 2*half_T_dim_prim]
        #print(f"----> D3 {D3.shape}")
        B, D, C, T3 = D3.shape
        D3 = D3.view(B, D, C*T3)
        
        
        x_bot1 = self.Bottom1(D3)   # [B, D, C*T3]
        x_bot1 = x_bot1.permute(0, 2, 1)
        #print(f"x_bot1 {x_bot1.shape}")
        
        # attention in Bottom stages
        pos_enc_2 = self.TimeSet_Pooling_2(absolute_PE, C)
        x_bot1_attn = self.TransformerEnc2(x_bot1, x_bot1, pos_enc_2, PE_object)  # [B, C*T3, D]
        #print(f"x_bot1_attn {x_bot1_attn.shape}")
        x_bot1_attn = x_bot1_attn.permute(0, 2, 1) # [B, D, C*T3]
        
        
        x_bot2 = self.Bottom2(x_bot1_attn) # [B, D, C*T3]
        x_bot2 = x_bot2.view(B, D, C, T3)
        #print(f"x_bot2 {x_bot2.shape}")

        U1 = self.Up1(x_bot2, D3)  
        B, D, C, T4 = U1.shape 
        #print(f"----> U1 {U1.shape}")
        U1 = U1.permute(0,2,3,1).view(B, C*T4, D)

        # attention between decoder stages
        U1_attn = self.TransformerEnc3(U1, U1, pos_enc_1, PE_object) # [B, C*T4, D]  
        aux_U1_attn = self.aux_head_up1(U1_attn).permute(0, 2, 1)
        #print(f"----> aux_U1_attn {aux_U1_attn.shape}")
        
        U1_attn = U1_attn.permute(0,2,1).view(B, D, C, T4)
        D2      =      D2.permute(0,2,1).view(B, D, C, T2)
        #print(f"----> T2 -- T4 {T2}, {T4}")
        U2 = self.Up2(U1_attn, D2)  
        B, D, C, T5 = U2.shape 
        #print(f"----> U2 {U2.shape}")
        #U2 = U2.permute(0,2,3,1).view(B, C*T5, D)
        aux_U2 = self.aux_head_up2(U2.permute(0,2,3,1).view(B, C*T5, D)).permute(0, 2, 1)
        #print(f"----> aux_u2 {aux_U2.shape}")

        U3_Dec1, U3_Dec2 = self.Up3(U2, list_D1)
        #print(f"----> U3_Dec1 {U3_Dec1.shape}")
        #print(f"----> U3_Dec2 {U3_Dec2.shape}")
        

        return U3_Dec1.squeeze(1), U3_Dec2.squeeze(1), aux_U2, aux_U1_attn

































