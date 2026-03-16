import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
import numpy as np
torch.autograd.set_detect_anomaly(True)





#########################################################################################################
#########################################################################################################
#########################################################################################################

#                    Self-Supervised Learning Architecture part Below !!!!!

#########################################################################################################
#########################################################################################################
#########################################################################################################

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
    
from einops import rearrange, repeat
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

        return [x1, x3] #x_concat



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

            nn.ConvTranspose1d(mid_dim, out_dim, kernel_size=65, stride=3, dilation=4, padding=127, output_padding=0) #, output_padding=
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
        "Take in model size and number of heads."
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
        #---------------------------------------------------------------------        if self.dropout is not None:
        
        #attn_weights = self.dropout(attn_weights)
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


class TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, pos_enc, PE_object):
        for layer in self.layers:
            x = layer(x, pos_enc, PE_object)
        return self.norm(x)


class Transformer_EncoderLayer(nn.Module):
    '''
    An encoder layer

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
        x = self.sublayer_output[0](query, lambda x: self.MultiH_Attn(query, x_2, x_1, pos_enc, PE_object))  #Encoder self-attention
        out = self.sublayer_output[1](x, self.feed_forward)

        return out

class Gate_Transformer_Layer(nn.Module):
    def __init__(self, MultiH_Attn, feed_forward, dim_emb, dropout):
        super(Gate_Transformer_Layer, self).__init__()
        self.MultiH_Attn = MultiH_Attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(dim_emb, dropout), 2)
        
        self.lin_1 = nn.Linear(dim_emb, dim_emb)

    def forward(self, x_1, x_2, pos_enc, PE_object, emb=None):
        
        query = self.lin_1(x_1)
        x = self.sublayer_output[0](query, lambda x: self.MultiH_Attn(query, x_2, x_2, pos_enc, PE_object))  #Encoder self-attention
        out = self.sublayer_output[1](x, self.feed_forward)

        return out
    

class PositionwiseFeedForward(nn.Module):

    def __init__(self, dim_emb, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_emb, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_emb)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))







class CoSup_UNet_SSL(nn.Module): 
    def __init__(self, nb_attn_heads_for_SSL):
        super(CoSup_UNet_SSL, self).__init__()


        self.TimeSet_Pooling_1 = TimeSet_Pooling_1()
        self.TimeSet_Pooling_2 = TimeSet_Pooling_2()
        
        dp_proba = 0.2
        
        attn = MultiHeadedAttention(nb_attn_heads_for_SSL, 64)
        ff = PositionwiseFeedForward(64, 128, dropout=dp_proba)
        attn2 = MultiHeadedAttention(nb_attn_heads_for_SSL, 128)
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
            nn.Linear(64, 19),  
        )
        self.aux_head_up1 = nn.Sequential(
            nn.Linear(64, 19),
        )
        
        

    def forward(self, input_x, absolute_PE, PE_object, mask=None):

        B, C, T = input_x.shape
        input_x = input_x.unsqueeze(1)
        

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

        # Up1      
        U1 = self.Up1(x_bot2, D3)  
        B, D, C, T4 = U1.shape 
        #print(f"----> U1 {U1.shape}")
        U1 = U1.permute(0,2,3,1).view(B, C*T4, D)

        # attention between decoder stages
        U1_attn = self.TransformerEnc3(U1, U1, pos_enc_1, PE_object) # [B, C*T4, D]  
        aux_U1_attn = self.aux_head_up1(U1_attn).permute(0, 2, 1)
        #print(f"----> aux_U1_attn {aux_U1_attn.shape}")
        
        # Up2    
        U1_attn = U1_attn.permute(0,2,1).view(B, D, C, T4)
        D2      =      D2.permute(0,2,1).view(B, D, C, T2)
        #print(f"----> T2 -- T4 {T2}, {T4}")
        U2 = self.Up2(U1_attn, D2)  
        B, D, C, T5 = U2.shape 
        #print(f"----> U2 {U2.shape}")
        #U2 = U2.permute(0,2,3,1).view(B, C*T5, D)
        aux_U2 = self.aux_head_up2(U2.permute(0,2,3,1).view(B, C*T5, D)).permute(0, 2, 1)
        #print(f"----> aux_u2 {aux_U2.shape}")

        # Up3
        U3_Dec1, U3_Dec2 = self.Up3(U2, list_D1)
        #print(f"----> U3_Dec1 {U3_Dec1.shape}")
        #print(f"----> U3_Dec2 {U3_Dec2.shape}")
        

        return U3_Dec1.squeeze(1), U3_Dec2.squeeze(1), aux_U2, aux_U1_attn


#########################################################################################################
#########################################################################################################
#########################################################################################################

#                    Self-Supervised Learning Architecture part Above !!!!!

#########################################################################################################
#########################################################################################################
#########################################################################################################

















#########################################################################################################
#########################################################################################################
#########################################################################################################

#                   MOE Finetuning Architecture part Below !!!!!

#########################################################################################################
#########################################################################################################
#########################################################################################################

class SSL_ExpertEncoder(nn.Module):

    def __init__(self, pretrained_ssl: nn.Module):
        super().__init__()

        self.Down1 = pretrained_ssl.Down1
        self.Down2 = pretrained_ssl.Down2
        self.Down3 = pretrained_ssl.Down3
        self.TransformerEnc1 = pretrained_ssl.TransformerEnc1
        self.TransformerEnc2 = pretrained_ssl.TransformerEnc2
        self.Bottom1 = pretrained_ssl.Bottom1
        self.Bottom2 = pretrained_ssl.Bottom2
        self.TimeSet_Pooling_1 = pretrained_ssl.TimeSet_Pooling_1
        self.TimeSet_Pooling_2 = pretrained_ssl.TimeSet_Pooling_2

    def forward(self, input_x, absolute_PE, PE_object, mask=None):
        B, C, T = input_x.shape
        x = input_x.unsqueeze(1)

        list_D1 = self.Down1(x)

        D2 = self.Down2(list_D1, C) 
        B, D, C, T2 = D2.shape
        D2 = D2.view(B, C*T2, D)

        pos_enc_1 = self.TimeSet_Pooling_1(absolute_PE, C)
        D2_attn = self.TransformerEnc1(D2, D2, pos_enc_1, PE_object)
        D2_attn = D2_attn.permute(0, 2, 1).view(B, D, C, T2)

        D3 = self.Down3(D2_attn)  # [B, D, C, T3]
        B, D, C, T3 = D3.shape
        D3 = D3.view(B, D, C*T3)

        x_bot1 = self.Bottom1(D3).permute(0, 2, 1)  # [B, C*T3, D]

        pos_enc_2 = self.TimeSet_Pooling_2(absolute_PE, C)
        x_bot1_attn = self.TransformerEnc2(x_bot1, x_bot1, pos_enc_2, PE_object)  # [B, C*T3, D]
        x_bot1_attn = x_bot1_attn.permute(0, 2, 1)  # [B, D, C*T3]

        x_bot2 = self.Bottom2(x_bot1_attn)  # [B, D, L]
        return x_bot2

class FeaturePool(nn.Module):
    def __init__(self, mode: str = "mean"):
        super().__init__()
        assert mode in ("mean", "max")
        self.mode = mode
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):  # [B, D, L]
        if self.mode == "mean":
            return x.mean(dim=-1)  # [B, D]
        else:
            return self.maxpool(x).squeeze(-1)  # [B, D]

class pooling_L_dim(nn.Module):
    def __init__(self, mode: str = "mean"):
        super().__init__()
        assert mode in ("mean", "max")
        self.mode = mode
        #self.maxpool = nn.AdaptiveMaxPool1d(1)
        if mode == "max":
            self.maxpool = nn.AdaptiveMaxPool2d((128,1))

    def forward(self, nb_channels, x):  # [B, D, L]
        B,D,L = x.shape
        x = x.reshape(B,nb_channels,D,-1)
        if self.mode == "mean":
            return x.mean(dim=-1).squeeze(-1)  # [B, C, D]
        elif self.mode == "max":
            #print(f"x {x.shape}")
            o = self.maxpool(x).squeeze(-1)  # [B, C, D]
            #print(f"o {o.shape}")
            return o
        

class GateNet(nn.Module):

    def __init__(self, dim_emb: int, hidden: int = 128, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.LayerNorm(3 * dim_emb),
            nn.Linear(3 * dim_emb, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, h_cat):  
        logits = self.net(h_cat)                
        w = torch.softmax(logits / self.temperature, dim=-1)
        return w, logits
    
class input_encoding_Network_paaaast(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, h):  # [B, C, L]
        return self.net(h)  # [B, C, D]
        
class input_encoding_Network(nn.Module):
    """[B,C,T] -> [B,C,E]"""
    def __init__(self, C: int, E: int, k: int = 16, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=k, padding=k//2, groups=C, bias=False),
            nn.BatchNorm1d(C),
            nn.GELU(),
            nn.Dropout(drop),

            nn.Conv1d(C, C, kernel_size=k, padding=k//2, groups=C, bias=False),
            nn.BatchNorm1d(C),
            nn.GELU(),
            nn.Dropout(drop),

            nn.AdaptiveAvgPool1d(E),   # [B,C,E]
        )

    def forward(self, x):
        return self.net(x)

class MLPClassifier(nn.Module):
    def __init__(self, dim_emb: int, num_classes: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_emb),
            nn.Linear(dim_emb, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, h):  # [B, D]
        return self.net(h)  # [B, num_classes]

        
class SSL_3Expert_GatedFusion(nn.Module):
    def __init__(
        self,
        expert1_ssl: nn.Module,
        expert2_ssl: nn.Module,
        expert3_ssl: nn.Module,
        dim_emb: int,
        num_classes: int,
        L_pool_mode: str = "mean",
        C_pool_mode: str = "mean",
        gate_hidden: int = 128,
        gate_temperature: float = 1.5,
        clf_hidden: int = 128,
        clf_dropout: float = 0.2,
        freeze_experts: str = "YES",
        normalize_expert_embeddings: str = "YES",
        finetuning_mode: int = 0,
        segment_Len_secs: int = 0,
        sampling_Freq: float = 0,
        nb_channels: int =1
    ):
        super().__init__()

        self.experts = nn.ModuleList([
            SSL_ExpertEncoder(expert1_ssl),
            SSL_ExpertEncoder(expert2_ssl),
            SSL_ExpertEncoder(expert3_ssl),
        ])
        
        sequence_dim = int(segment_Len_secs*sampling_Freq)
        
        self.finetuning_mode = finetuning_mode
        self.sampling_Freq = sampling_Freq
        
        if segment_Len_secs >= 2:
            frequency_dim = sampling_Freq + 1
            self.nperseg  = int(sampling_Freq * 2)
            self.noverlap = int(sampling_Freq)
        elif segment_Len_secs == 1:
            frequency_dim = (sampling_Freq//2) + 1
            self.nperseg  = int(sampling_Freq)
            self.noverlap = 0
        else:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Sequence length should be > 1 sec")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        if self.finetuning_mode == 0:
            #self.pool = FeaturePool(mode=pool_mode)
            self.pooling_L = pooling_L_dim(mode=pool_mode)
            self.pooling_C = nn.AdaptiveAvgPool1d(1)

            self.emb_ln = nn.LayerNorm(dim_emb) if normalize_expert_embeddings=="YES" else nn.Identity()

            self.gate = GateNet(dim_emb=dim_emb, hidden=gate_hidden, temperature=gate_temperature)
            self.classifier = MLPClassifier(dim_emb=dim_emb, num_classes=num_classes, hidden=clf_hidden, dropout=clf_dropout)
        elif self.finetuning_mode == 1:
            self.gate_LinearT_1 = nn.Linear(frequency_dim, dim_emb)
            self.gate_LinearT_2 = nn.Linear(frequency_dim, dim_emb)
            self.gate_LinearT_3 = nn.Linear(frequency_dim, dim_emb)
            self.gate_activation = nn.Sigmoid()
            self.pooling_L = pooling_L_dim(mode=L_pool_mode)
            if C_pool_mode == "max":
                self.pooling_C = nn.AdaptiveMaxPool1d(1)
            else:
                self.pooling_C = nn.AdaptiveAvgPool1d(1)
            self.classifier = MLPClassifier(dim_emb=dim_emb*3, num_classes=num_classes, hidden=clf_hidden*3, dropout=clf_dropout)
        elif self.finetuning_mode == 2:
            self.input_encoding_Net = input_encoding_Network(nb_channels,dim_emb)
            self.gate_LinearT_1 = nn.Linear(dim_emb, dim_emb)
            self.gate_LinearT_2 = nn.Linear(dim_emb, dim_emb)
            self.gate_LinearT_3 = nn.Linear(dim_emb, dim_emb)
            self.gate_activation = nn.Sigmoid()
            self.pooling_L = pooling_L_dim(mode=pool_mode)
            self.pooling_C = nn.AdaptiveAvgPool1d(1)
            self.classifier = MLPClassifier(dim_emb=dim_emb*3, num_classes=num_classes, hidden=clf_hidden*3, dropout=clf_dropout)

        if freeze_experts=="YES":
            for e in self.experts:
                for p in e.parameters():
                    p.requires_grad = False

    def unfreeze_experts(self):
        for e in self.experts:
            for p in e.parameters():
                p.requires_grad = True

    def forward(self, input_x, absolute_PE, PE_object, mask=None):
        
        x_bot2_list = [e(input_x, absolute_PE, PE_object, mask) for e in self.experts]  # 3 x [B,D,L]
        B,C,T = input_x.shape

        if self.finetuning_mode == 0:
            h_list = [self.emb_ln(self.pooling_L(C, xb)) for xb in x_bot2_list]  # 3 x [B,C,D]
            
            h_cat = torch.cat(h_list, dim=-1)  # [B, C, 3D]
            w, gate_logits = self.gate(h_cat) # w: [B,C,3]

            h_fused = (
                w[:, :, 0:1] * h_list[0] +
                w[:, :, 1:2] * h_list[1] +
                w[:, :, 2:3] * h_list[2]
            )  # [B,D]

            h_fused = h_fused.permute(0,2,1)
            h_fused = self.pooling_C(h_fused).squeeze()
            
            logits = self.classifier(h_fused)  # [B,num_classes]
            return logits, w, gate_logits
        
        elif self.finetuning_mode == 1:
            # ===== Compute the Power spectrum of the raw input =====
            psd_x = welch_psd(input_x, fs=self.sampling_Freq, nperseg=self.nperseg, noverlap=self.noverlap)
            #print(f"----> psd_x {psd_x.shape}")

            # ===== Gating Tensors =====
            gate_tensor_1 = self.gate_activation(self.gate_LinearT_1(psd_x))
            gate_tensor_2 = self.gate_activation(self.gate_LinearT_2(psd_x))
            gate_tensor_3 = self.gate_activation(self.gate_LinearT_3(psd_x))

            # ===== Pooling the L dimension from the Experts output  =====
            x_bot2_list = [self.pooling_L(C, xb) for xb in x_bot2_list]

            # ===== Experts Gated merge  =====
            h_fused = torch.cat([x_bot2_list[0]*gate_tensor_1 , x_bot2_list[1]*gate_tensor_2 , x_bot2_list[2]*gate_tensor_3], dim=-1)

            h_fused = h_fused.permute(0,2,1)
            h_fused = self.pooling_C(h_fused).squeeze()
            #print(f"----> h_fused 2 {h_fused.shape}")

            logits = self.classifier(h_fused)  # [B,num_classes]
            return logits


        elif self.finetuning_mode == 2:
            # ===== Compute the Power spectrum of the raw input =====
            x_enc_for_gating = self.input_encoding_Net(input_x)
            

            # ===== Pooling the L dimension from the Experts output  =====
            x_bot2_list = [self.pooling_L(C, xb) for xb in x_bot2_list]

            # ===== Gating Tensors =====
            #expert_1_out_for_gating = x_bot2_list[0].reshape(B,C,-1)
            gate_tensor_1 = self.gate_activation(self.gate_LinearT_1(x_enc_for_gating)) #expert_1_out_for_gating
            gate_tensor_2 = self.gate_activation(self.gate_LinearT_2(x_enc_for_gating))
            gate_tensor_3 = self.gate_activation(self.gate_LinearT_3(x_enc_for_gating))

            # ===== Experts Gated merge  =====
            #h_fused = x_bot2_list[0]*gate_tensor_1 + x_bot2_list[1]*gate_tensor_2 + x_bot2_list[2]*gate_tensor_3
            h_fused = torch.cat([x_bot2_list[0]*gate_tensor_1 , x_bot2_list[1]*gate_tensor_2 , x_bot2_list[2]*gate_tensor_3], dim=-1)

            h_fused = h_fused.permute(0,2,1)
            h_fused = self.pooling_C(h_fused).squeeze()
            #print(f"----> h_fused 2 {h_fused.shape}")

            logits = self.classifier(h_fused)  # [B,num_classes]
            return logits


#########################################################################################################
#########################################################################################################
#########################################################################################################

#                   MOE Finetuning Architecture part Above !!!!!

#########################################################################################################
#########################################################################################################
#########################################################################################################

def welch_psd(
    x: torch.Tensor,
    fs: float,
    nperseg: int,
    noverlap: int = None,
    window: str = "hann",
    detrend: bool = True,
    eps: float = 1e-12,
):

    assert x.ndim == 3, "x must be [B, C, L]"
    B, C, L = x.shape


    if noverlap is None:
        noverlap = nperseg // 2
    assert 0 <= noverlap < nperseg <= L

    if detrend:
        x = x - x.mean(dim=-1, keepdim=True)

    step = nperseg - noverlap
    nseg = 1 + (L - nperseg) // step

    # [B, C, nseg, nperseg]
    frames = x.unfold(dimension=-1, size=nperseg, step=step)

    # window
    if window == "hann":
        w = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    elif window == "hamming":
        w = torch.hamming_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    else:
        raise ValueError("window must be 'hann' or 'hamming'")

    frames = frames * w.view(1, 1, 1, -1)

    # rFFT: [B, C, nseg, F]
    X = torch.fft.rfft(frames, dim=-1)
    Pxx = (X.real**2 + X.imag**2)

    Pxx = Pxx / (fs * (w.pow(2).sum()) + eps)

    # average across segments -> [B, C, F]
    psd = Pxx.mean(dim=2)

    return psd
