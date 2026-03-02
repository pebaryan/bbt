"""Mamba-based language model (MambaMLM) - a BitByte variant."""

import math
import torch
import torch.nn as nn

from .mamba.mamba_block import MambaBlock


class MambaMLM(nn.Module):
    """Mamba-based language model using BitByte quantization.

    Implements a Mamba-based language model with:
    - BitLinear layers in SSM and MLP
    - Ternary quantization with STE
    - RMSNorm
    """

    def __init__(
        self,
        vocab_size: int = 256,
        n_layer: int = 24,
        d_model: int = 1536,
        d_ff: int = 4096,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        act_quant: bool = True,
        use_checkpoint: bool = True,
        time_step_min: float = 1e-3,
        time_step_max: float = 1e-1,
        dt_init: str = "log_uniform",
        a_init: str = "uniform_0_16",
    ):
        """Initialize MambaMLM.

        Args:
            vocab_size: Size of vocabulary (default 256 for bytes)
            n_layer: Number of Mamba blocks
            d_model: Model dimension
            d_ff: Feed-forward dimension
            d_state: State dimension for SSM
            d_conv: Convolution kernel size for SSM
            expand: Expansion factor for SSM inner dimension
            act_quant: Whether to quantize activations
            use_checkpoint: Whether to use gradient checkpointing for SSM
            time_step_min: Minimum initial dt value for delta softplus bias
            time_step_max: Maximum initial dt value for delta softplus bias
            dt_init: Initialization for delta bias ("log_uniform" or "zeros")
            a_init: Initialization for A_log ("uniform_0_16" or "log_arange")
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Lower init scale keeps logits in a sane range for fp16.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    d_ff,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    act_quant=act_quant,
                    use_checkpoint=use_checkpoint,
                    time_step_min=time_step_min,
                    time_step_max=time_step_max,
                    dt_init=dt_init,
                    a_init=a_init,
                )
                for _ in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            idx: Input token indices [B, T]

        Returns:
            Logits [B, T, vocab_size]
        """
        # idx: [B,T] bytes
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B,T,256]
        return logits
