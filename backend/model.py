"""
APMAN-Net v2.0: Adaptive Physics-Guided Multi-Stage Attention Network for Underwater Image Enhancement
Model Architecture Module
Authors: Madala Leela Venkata Siva Spoorthi, Bhargav Gopi Anagani, Ramagiri Vardhan, Paturu Venkat Sujay Reddy
Guide: Dr. Anil Vitthalrao Turukmane, Head, Dept. of Security, Professor Grade 2, SCOPE, VIT-AP University
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedFilter(nn.Module):
    """
    Edge-preserving guided filter (He et al., TPAMI 2013)
    Used to refine the transmission map t(x) using the color-compensated image as guide.
    """
    def __init__(self, r=8, eps=1e-4):
        super().__init__()
        self.r = r
        self.eps = eps

    def mean_filter(self, x):
        k = 2 * self.r + 1
        x_pad = F.pad(x, (self.r, self.r, self.r, self.r), mode='reflect')
        return F.avg_pool2d(x_pad, kernel_size=k, stride=1)

    def forward(self, guide, src):
        g = guide.mean(dim=1, keepdim=True)
        mi = self.mean_filter(g)
        mp = self.mean_filter(src)
        mip = self.mean_filter(g * src)
        mii = self.mean_filter(g * g)
        a = (mip - mi * mp) / (mii - mi * mi + self.eps)
        b = mp - a * mi
        return self.mean_filter(a) * g + self.mean_filter(b)


class WaterConditionAnalyzer(nn.Module):
    """
    Stage 1: Water Severity Index (WSI)
    Extracts 6 physics-based optical degradation statistics:
    - Blue channel ratio
    - Green channel ratio
    - Mean luminance (brightness)
    - Contrast (luminance std)
    - Saturation (max - min)
    - Mean gradient magnitude (detail/blurriness)
    """
    def forward(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        tot = r.mean([-1, -2]) + g.mean([-1, -2]) + b.mean([-1, -2]) + 1e-6
        blue_ratio = b.mean([-1, -2]) / tot
        green_ratio = g.mean([-1, -2]) / tot
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        brightness = lum.mean([-1, -2])
        contrast = torch.sqrt((lum - lum.mean([-1, -2], keepdim=True)).pow(2).mean([-1, -2]) + 1e-6)
        saturation = (x.max(dim=1)[0] - x.min(dim=1)[0]).mean([-1, -2])
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean([-1, -2, -3])
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean([-1, -2, -3])
        return torch.stack([blue_ratio, green_ratio, brightness, contrast, saturation, (dx + dy) / 2.0], dim=1)


class AdaptiveColorCompensation(nn.Module):
    """
    Stage 2: Adaptive Color Compensation
    3-layer MLP that takes 6 WSI statistics and predicts dynamic channel gains and biases.
    Bounded with tanh and zero-initialized last layer for stability.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 6)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, wsi):
        p = self.mlp(wsi)
        gain = 1.0 + torch.tanh(p[:, :3].view(-1, 3, 1, 1)) * 0.5
        bias = torch.tanh(p[:, 3:].view(-1, 3, 1, 1)) * 0.2
        return (gain * x + bias).clamp(0.0, 1.0), gain, bias


class UNetTransmission(nn.Module):
    """
    Stage 3: Deep Lightweight U-Net for Medium Transmission Map Estimation
    Estimates the optical medium transmission t(x) in [0, 1].
    """
    def __init__(self):
        super().__init__()
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU()
            )
        self.inc = double_conv(3, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(32, 64))

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up1 = double_conv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up2 = double_conv(32, 16)

        self.outc = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u1 = self.conv_up1(torch.cat([x2, self.up1(x3)], dim=1))
        u2 = self.conv_up2(torch.cat([x1, self.up2(u1)], dim=1))
        return self.outc(u2)


class MultiScaleFeatureExtractor(nn.Module):
    """
    Stage 4: Multi-Scale Feature Extraction (MSFE)
    Parallel receptive fields:
    - 3x3 conv: captures fine details and localized textures
    - 5x5 conv: captures regional boundary structures
    - Dilated 3x3 (rate=2): captures broad contextual semantics without downsampling
    Fuses 96 channels to 128 channels via 1x1 conv.
    """
    def __init__(self, in_ch=3, branch_ch=32, out_ch=128):
        super().__init__()
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 5, padding=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        self.bd = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * branch_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.b3(x), self.b5(x), self.bd(x)], dim=1))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018)
    Sequential Channel Attention + Spatial Attention.
    """
    def __init__(self, channels=128, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        ca = self.sig(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        x = x * ca
        sa = self.sa(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1))
        return x * sa


class ECABlock(nn.Module):
    """
    Efficient Channel Attention (Wang et al., CVPR 2020)
    Replaces costly FC layers with fast 1D convolution over channel neighbors.
    """
    def __init__(self, ch=128):
        super().__init__()
        k = int(abs(math.log2(ch) / 2 + 0.5))
        k = k if k % 2 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(self.pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))


class EdgeAttentionModule(nn.Module):
    """
    Edge-Guided Attention Module
    Extracts Sobel (sx, sy) and Laplacian high-frequency edge gradients to gate deep features.
    """
    def __init__(self, ch=128):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sx.transpose(-1, -2).contiguous())
        self.register_buffer('lap', torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.gate = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feats, guide):
        g = guide.mean(1, keepdim=True)
        edge = ((F.conv2d(g, self.sx, padding=1)**2 + F.conv2d(g, self.sy, padding=1)**2 + 1e-6).sqrt() + F.conv2d(g, self.lap, padding=1).abs()) / 2
        B = edge.shape[0]
        mn = edge.flatten(1).min(1)[0].view(B, 1, 1, 1)
        mx = edge.flatten(1).max(1)[0].view(B, 1, 1, 1)
        normalized_edge = (edge - mn) / torch.clamp(mx - mn, min=1e-6)
        return feats * self.gate(normalized_edge), normalized_edge


class ConfidenceMapPredictor(nn.Module):
    """
    Stage 6 & 7: Confidence Prediction and Fusion Network
    Predicts a per-pixel confidence map C in [0, 1].
    Outputs fused image = C * i_enh + (1 - C) * x_dehaze, preventing over-enhancement in uncertain regions.
    """
    def __init__(self, ch=128):
        super().__init__()
        self.enh = nn.Sequential(
            nn.Conv2d(ch + 3, 64, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid()
        )
        self.conf = nn.Sequential(
            nn.Conv2d(ch + 3, 64, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feats, x_orig):
        cat = torch.cat([feats, x_orig], dim=1)
        i_enh = self.enh(cat)
        c = self.conf(cat)
        fused = c * i_enh + (1.0 - c) * x_orig
        return fused, c, i_enh


class ResidualRefinementNetwork(nn.Module):
    """
    Stage 8: Residual Refinement Network
    3 consecutive residual blocks with BatchNorm and GELU activations.
    Corrects subtle color artifacts and fine boundary gradients.
    """
    def __init__(self):
        super().__init__()
        def res_block(ch):
            return nn.Sequential(
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            )

        class RB(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self.n = res_block(ch)
            def forward(self, x):
                return x + self.n(x)

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            RB(64),
            RB(64),
            RB(64),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class APMANNet(nn.Module):
    """
    Full APMAN-Net v2.0 Architecture
    Parameters: ~0.52 Million
    """
    def __init__(self):
        super().__init__()
        self.analyzer = WaterConditionAnalyzer()
        self.color = AdaptiveColorCompensation()
        self.haze = UNetTransmission()
        self.gfilter = GuidedFilter()
        self.mscale = MultiScaleFeatureExtractor()
        self.cbam = CBAM(128)
        self.eca = ECABlock(128)
        self.edge = EdgeAttentionModule(128)
        self.conf = ConfidenceMapPredictor(128)
        self.refiner = ResidualRefinementNetwork()
        self.t_min = 0.15

    def forward(self, x, return_intermediates=False):
        # 1. Water Condition Analysis
        wsi = self.analyzer(x)

        # 2. Adaptive Color Compensation
        x_cc, gains, biases = self.color(x, wsi)

        # 3. Physics-Guided Transmission & Dehazing
        t_raw = self.haze(x_cc)
        t = self.gfilter(x_cc, t_raw).clamp(self.t_min, 1.0)
        B, C, H, W = x_cc.shape
        flat = x_cc.view(B, C, -1)
        A = flat.topk(max(1, int(flat.shape[-1] * 0.001)), dim=-1)[0].mean(-1).view(B, C, 1, 1)
        x_dh = torch.clamp((x_cc - A * (1.0 - t)) / (t + 1e-6), 0.0, 1.0)
        x_dehaze = 0.7 * x_dh + 0.3 * x_cc

        # 4 & 5. Multi-Scale Attention Features
        mscale_feats = self.mscale(x_dehaze)
        cbam_feats = self.cbam(mscale_feats)
        eca_feats = self.eca(cbam_feats)
        feats, edge_map = self.edge(eca_feats, x_cc)

        # 6 & 7. Confidence Prediction & Fusion
        fused, conf_map, i_enh = self.conf(feats, x_dehaze)

        # 8. Residual Refinement
        out = self.refiner(fused)

        result = {"output": out}
        if return_intermediates:
            result.update({
                "wsi": wsi,
                "gains": gains,
                "biases": biases,
                "color_compensated": x_cc,
                "transmission_raw": t_raw,
                "transmission": t,
                "ambient_light": A,
                "dehazed": x_dehaze,
                "edge_map": edge_map,
                "confidence_map": conf_map,
                "i_enh": i_enh,
                "fused": fused
            })
        return result
