""" Pytorch implementation of HRNet, a neural network for multi-frame super resolution (MFSR) by recursive fusion. """
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .cbam import CBAM


class ResidualBlock(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        return x + self.block(x)


class FusionAttention(nn.Module):
    """
    Cross-frame attention for selective fusion inside RecuversiveNet.

    At each recursive fusion step, the blind model concatenates alice+bob
    and passes them through a ResidualBlock.  This module instead asks:
    "at each spatial position (x,y), which frame is more trustworthy?"
    and produces a weighted blend before the residual correction.

    Starts as a passthrough (all weights zero, blend=0) so it has zero
    effect at epoch 1 and cannot destabilise a loaded pretrained weight.
    The blend gate is a single learnable scalar — if the attention learns
    nothing useful, it stays near 0 and the original fuse path dominates.
    """

    def __init__(self, channels):
        super(FusionAttention, self).__init__()
        reduced = max(1, channels // 4)   # 64 -> 16 channels for Q/K

        # Query: "what is alice looking for?"
        self.query = nn.Conv2d(channels, reduced, kernel_size=1, bias=True)
        # Key:   "what does bob offer?"
        self.key   = nn.Conv2d(channels, reduced, kernel_size=1, bias=True)
        # Values: full-channel projections used in the weighted blend
        self.value_alice = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.value_bob   = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # Blend gate: sigmoid(0) = 0.5, but since values start at 0
        # the attended output is also 0, so effective contribution is 0.
        # Grows as training proceeds and attention becomes useful.
        self.blend = nn.Parameter(torch.full((1,), -6.0))

        self._init_passthrough()

    def _init_passthrough(self):
        """Zero all weights so this module is a strict identity at epoch 1."""
        for m in [self.query, self.key, self.value_alice, self.value_bob]:
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, alice, bob):
        """
        Args:
            alice: [B, C, H, W]
            bob:   [B, C, H, W]
        Returns:
            blended: [B, C, H, W]  attention-guided combination
        """
        # Per-position similarity: where do alice and bob agree?
        score  = torch.sum(self.query(alice) * self.key(bob), dim=1, keepdim=True)  # [B,1,H,W]
        weight = torch.sigmoid(score)   # high -> trust bob more here

        # Weighted blend of value projections
        attended = (1.0 - weight) * self.value_alice(alice) + weight * self.value_bob(bob)

        # Blend gate: how much does attention contribute overall?
        # sigmoid(self.blend) starts at 0.5 but attended starts at 0,
        # so output = alice until the network learns to use attention.
        alpha   = torch.sigmoid(self.blend)
        return alpha * attended + (1.0 - alpha) * alice


class Encoder(nn.Module):
    def __init__(self, config, use_cbam=False, cbam_reduction=8, cbam_spatial_kernel=7,
                 use_checkpointing=False):
        super(Encoder, self).__init__()
        in_channels  = config["in_channels"]
        num_layers   = config["num_layers"]
        kernel_size  = config["kernel_size"]
        channel_size = config["channel_size"]
        padding      = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size,
                      kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )
        self.res_layers = nn.Sequential(
            *[ResidualBlock(channel_size, kernel_size) for _ in range(num_layers)]
        )
        final_layers = [
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size,
                      kernel_size=kernel_size, padding=padding)
        ]
        if use_cbam:
            final_layers.append(
                CBAM(channels=channel_size, reduction=cbam_reduction,
                     spatial_kernel=cbam_spatial_kernel)
            )
        self.final = nn.Sequential(*final_layers)
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        x = self.init_layer(x)
        if self.use_checkpointing and self.training:
            x = checkpoint(self.res_layers, x)
            x = checkpoint(self.final, x)
        else:
            x = self.res_layers(x)
            x = self.final(x)
        return x


class RecuversiveNet(nn.Module):
    def __init__(self, config):
        super(RecuversiveNet, self).__init__()
        self.input_channels = config["in_channels"]
        self.num_layers     = config["num_layers"]
        self.alpha_residual = config["alpha_residual"]
        self.use_checkpointing = bool(config.get("use_checkpointing", False))
        self.use_fusion_attention = bool(config.get("use_fusion_attention", True))
        kernel_size         = config["kernel_size"]
        padding             = kernel_size // 2

        # Original blind fusion — kept as a residual corrector
        self.fuse = nn.Sequential(
            ResidualBlock(2 * self.input_channels, kernel_size),
            nn.Conv2d(in_channels=2 * self.input_channels, out_channels=self.input_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

        # Attention-guided fusion — shared across all recursion levels
        self.fusion_attn = FusionAttention(self.input_channels)

    def _run_fusion_attn(self, alice_flat, bob_flat):
        return self.fusion_attn(alice_flat, bob_flat)

    def forward(self, x, alphas):
        """
        Args:
            x      : tensor (B, L, C, W, H)  encoded frames
            alphas : tensor (B, L, 1, 1, 1)  validity mask
        Returns:
            out    : tensor (B, C, W, H)      fused representation
        """
        batch_size, nviews, channels, width, height = x.shape
        parity   = nviews % 2
        half_len = nviews // 2

        while half_len > 0:
            alice = x[:, :half_len]                      # (B, L/2, C, H, W)
            bob   = x[:, half_len:nviews - parity]       # (B, L/2, C, H, W)
            bob   = torch.flip(bob, [1])

            # --- residual correction from original blind fuse (kept) ---
            alice_and_bob = torch.cat([alice, bob], 2).view(-1, 2 * channels, width, height)
            if self.use_checkpointing and self.training:
                residual = checkpoint(self.fuse, alice_and_bob)
            else:
                residual = self.fuse(alice_and_bob)
            residual      = residual.view(batch_size, half_len, channels, width, height)

            if self.use_fusion_attention:
                # --- attention-guided blend (new) ---
                alice_flat   = alice.reshape(-1, channels, width, height)
                bob_flat     = bob.reshape(-1, channels, width, height)
                if self.use_checkpointing and self.training:
                    attended_flat = checkpoint(self._run_fusion_attn, alice_flat, bob_flat)
                else:
                    attended_flat = self.fusion_attn(alice_flat, bob_flat)
                attended = attended_flat.view(batch_size, half_len, channels, width, height)
                # Attention leads; blind residual provides a small safety correction
                x = attended + 0.1 * residual
            else:
                # Legacy behavior: no fusion attention path.
                x = residual

            if self.alpha_residual:
                alphas_alice = alphas[:, :half_len]
                alphas_bob   = alphas[:, half_len:nviews - parity]
                alphas_bob   = torch.flip(alphas_bob, [1])
                x      = alice + alphas_bob * x
                alphas = alphas_alice

            nviews   = half_len
            parity   = nviews % 2
            half_len = nviews // 2

        return torch.mean(x, 1)


class Decoder(nn.Module):
    def __init__(self, config, use_cbam=False, cbam_reduction=8, cbam_spatial_kernel=7,
                 use_checkpointing=False):
        super(Decoder, self).__init__()

        if "upsample" in config and "conv" in config:
            upsample_cfg = config["upsample"]
            conv_cfg     = config["conv"]
            mode         = upsample_cfg.get("mode", "bilinear")
            align_corners = False if mode in ("bilinear", "bicubic", "trilinear") else None
            self.deconv  = nn.Sequential(
                nn.Upsample(scale_factor=upsample_cfg.get("scale_factor", 2),
                            mode=mode, align_corners=align_corners),
                nn.Conv2d(in_channels=conv_cfg["in_channels"],
                          out_channels=conv_cfg["out_channels"],
                          kernel_size=conv_cfg.get("kernel_size", 3),
                          padding=conv_cfg.get("kernel_size", 3) // 2),
                nn.PReLU(),
            )
            cbam_channels = conv_cfg["out_channels"]
        else:
            # Backward-compatible fallback for old ConvTranspose2d configs
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=config["deconv"]["in_channels"],
                    out_channels=config["deconv"]["out_channels"],
                    kernel_size=config["deconv"]["kernel_size"],
                    stride=config["deconv"]["stride"],
                    padding=config["deconv"].get("padding", 0),
                ),
                nn.PReLU(),
            )
            cbam_channels = config["deconv"]["out_channels"]

        self.cbam = None
        if use_cbam:
            self.cbam = CBAM(channels=cbam_channels, reduction=cbam_reduction,
                             spatial_kernel=cbam_spatial_kernel)
        self.use_checkpointing = use_checkpointing

        self.final = nn.Conv2d(
            in_channels=config["final"]["in_channels"],
            out_channels=config["final"]["out_channels"],
            kernel_size=config["final"]["kernel_size"],
            padding=config["final"]["kernel_size"] // 2
        )

    def forward(self, x):
        if self.use_checkpointing and self.training:
            x = checkpoint(self.deconv, x)
            if self.cbam is not None:
                x = checkpoint(self.cbam, x)
            x = checkpoint(self.final, x)
        else:
            x = self.deconv(x)
            if self.cbam is not None:
                x = self.cbam(x)
            x = self.final(x)
        return x


class HRNet(nn.Module):
    """HRNet: multi-frame super resolution by recursive fusion."""

    def __init__(self, config):
        super(HRNet, self).__init__()
        use_cbam          = config.get("use_cbam", False)
        cbam_reduction    = config.get("cbam_reduction", 8)
        cbam_spatial_kernel = config.get("cbam_spatial_kernel", 7)
        use_checkpointing = bool(config.get("use_checkpointing", False))
        recursive_cfg = dict(config["recursive"])
        if "use_checkpointing" not in recursive_cfg:
            recursive_cfg["use_checkpointing"] = use_checkpointing

        self.encode = Encoder(config["encoder"], use_cbam=use_cbam,
                              cbam_reduction=cbam_reduction,
                              cbam_spatial_kernel=cbam_spatial_kernel,
                              use_checkpointing=use_checkpointing)
        self.fuse   = RecuversiveNet(recursive_cfg)
        self.decode = Decoder(config["decoder"], use_cbam=use_cbam,
                              cbam_reduction=cbam_reduction,
                              cbam_spatial_kernel=cbam_spatial_kernel,
                              use_checkpointing=use_checkpointing)

    def forward(self, lrs, alphas):
        """
        Args:
            lrs    : tensor (B, L, H, W)
            alphas : tensor (B, L)
        Returns:
            srs    : tensor (B, 1, 2H, 2W)
        """
        batch_size, seq_len, height, width = lrs.shape
        lrs    = lrs.view(-1, seq_len, 1, height, width)
        alphas = alphas.view(-1, seq_len, 1, 1, 1)

        refs, _ = torch.median(lrs[:, :9], 1, keepdim=True)
        refs    = refs.expand(-1, seq_len, -1, -1, -1)

        stacked_input = torch.cat([lrs, refs], 2)                         # (B, L, 2, H, W)
        stacked_input = stacked_input.view(batch_size * seq_len, 2, width, height)

        layer1 = self.encode(stacked_input)                               # (B*L, C, H, W)
        layer1 = layer1.view(batch_size, seq_len, -1, width, height)      # (B, L, C, H, W)

        recursive_layer = self.fuse(layer1, alphas)                       # (B, C, H, W)
        srs             = self.decode(recursive_layer)                    # (B, 1, 2H, 2W)
        return srs