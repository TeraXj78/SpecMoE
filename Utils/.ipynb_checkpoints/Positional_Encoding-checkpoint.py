import torch
from einops import rearrange, repeat


class RotaryPE:

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    # Rotary frequency-based positional encoding
    def freq_pos_enc(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float, device=self.device) / dim)
        )  # (dim/2,)

        if positions.ndim > 2:
            positions = positions.squeeze(1)

        freq_pos = torch.einsum("BL, E -> BLE", positions, inv_freq)

        return repeat(freq_pos, "... E -> ... (E r)", r=2)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:

        x = rearrange(x, "... (E r) -> ... E r", r=2)
        x1, x2 = x.unbind(dim=-1)
        rotated = torch.stack((-x2, x1), dim=-1)
        return rearrange(rotated, "... E r -> ... (E r)")

    # RoPE rotation
    def rotate(self, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:

        while pos_enc.ndim < x.ndim:
            pos_enc = pos_enc.unsqueeze(1)  # [B, 1, L, D_head]

        cos_part = pos_enc.cos()
        sin_part = pos_enc.sin()
        return (x * cos_part) + (self.rotate_half(x) * sin_part)


