"""
Fake Dataset Factory — Gradio App
Synthetic Medical Chest X-Ray Generation

A clean, modern interface for generating and evaluating synthetic chest X-rays.
"""

import os

# Apple Silicon can run most PyTorch inference on Metal. Unsupported MPS
# operations fall back to CPU instead of crashing the local app.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
import math

# conditional imports
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False

try:
    from diffusers import (
        DDIMScheduler,
        DDPMScheduler,
        StableDiffusionImg2ImgPipeline,
        UNet2DModel,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "notebooks" / "outputs"

IMG_SIZE = 64
NOISE_DIM = 100
FEATURE_G = 64


# =============================================================================
# Custom CSS for a clean, modern look
# =============================================================================

CUSTOM_CSS = """
:root {
    --navy-950: #071426;
    --navy-900: #0b1f36;
    --navy-800: #15324f;
    --blue-700: #175cd3;
    --blue-600: #2474e5;
    --blue-100: #dcecff;
    --blue-50: #f0f7ff;
    --cyan-500: #18a6b8;
    --ink: #152437;
    --body: #4d6075;
    --muted: #718196;
    --line: #dce4ed;
    --line-strong: #c8d4e1;
    --canvas: #f4f7fb;
    --surface: #ffffff;
    --success: #14804a;
    --success-bg: #eaf8f0;
    --warning: #9a6700;
    --warning-bg: #fff7df;
    --radius-sm: 10px;
    --radius-md: 16px;
    --radius-lg: 22px;
    --shadow-sm: 0 1px 2px rgba(7, 20, 38, .04), 0 6px 18px rgba(7, 20, 38, .04);
    --shadow-md: 0 18px 45px rgba(7, 20, 38, .09);
}
body, .gradio-container { background: var(--canvas) !important; color: var(--ink) !important; }
.gradio-container {
    max-width: 1460px !important;
    margin: 0 auto !important;
    padding: 0 24px 48px !important;
}
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 72px;
    margin: 0 -24px;
    padding: 0 32px;
    color: white;
    background: var(--navy-950);
    border-bottom: 1px solid rgba(255,255,255,.09);
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    border-radius: 11px;
    color: white;
    background: var(--blue-600);
    font-size: 18px;
    font-weight: 800;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.22);
}
.brand-copy strong { display: block; font-size: 15px; letter-spacing: -.01em; }
.brand-copy span { color: #a9bdd0; font-size: 11px; }
.header-badges { display: flex; align-items: center; gap: 8px; }
.header-badge {
    padding: 7px 11px;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 999px;
    color: #c7d5e3;
    background: rgba(255,255,255,.055);
    font-size: 11px;
    font-weight: 650;
}
.header-badge.live::before { content: ''; display: inline-block; width: 7px; height: 7px; margin-right: 7px; border-radius: 50%; background: #36d17e; }
.product-hero {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr);
    gap: 40px;
    align-items: center;
    margin: 24px 0 20px;
    padding: 42px 48px;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    background: var(--surface);
    box-shadow: var(--shadow-sm);
}
.hero-kicker { color: var(--blue-700); font-size: 12px; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
.product-hero h1 { margin: 10px 0 14px !important; color: var(--navy-950); font-size: clamp(36px, 5vw, 62px) !important; line-height: 1.01 !important; letter-spacing: -.048em !important; }
.product-hero p { max-width: 720px; margin: 0 !important; color: var(--body); font-size: 17px; line-height: 1.65; }
.hero-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
.hero-chip { padding: 8px 12px; border-radius: 999px; color: var(--navy-800); background: var(--blue-50); border: 1px solid var(--blue-100); font-size: 12px; font-weight: 700; }
.hero-chip.caution { color: var(--warning); background: var(--warning-bg); border-color: #f3dfa3; }
.scan-visual { position: relative; display: grid; place-items: center; min-height: 270px; }
.scan-frame { width: min(100%, 310px); padding: 14px; border-radius: 20px; background: var(--navy-950); box-shadow: 0 24px 55px rgba(7,20,38,.23); transform: rotate(1.5deg); }
.scan-topline { display: flex; justify-content: space-between; margin-bottom: 10px; color: #8ca8c2; font: 600 9px/1 monospace; letter-spacing: .08em; }
.scan-frame svg { display: block; width: 100%; border-radius: 12px; background: #10263d; }
.scan-tag { position: absolute; right: 4%; bottom: 6%; padding: 9px 12px; color: var(--navy-900); background: white; border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow-md); font-size: 11px; font-weight: 750; }
.workflow-strip { display: grid; grid-template-columns: repeat(3, 1fr); margin-bottom: 22px; border: 1px solid var(--line); border-radius: var(--radius-md); background: var(--surface); box-shadow: var(--shadow-sm); }
.workflow-step { display: flex; gap: 13px; padding: 18px 20px; }
.workflow-step + .workflow-step { border-left: 1px solid var(--line); }
.step-number { display: grid; place-items: center; flex: 0 0 28px; height: 28px; border-radius: 8px; color: var(--blue-700); background: var(--blue-50); font-size: 12px; font-weight: 800; }
.workflow-step strong { display: block; color: var(--ink); font-size: 13px; }
.workflow-step span { display: block; margin-top: 2px; color: var(--muted); font-size: 11px; line-height: 1.4; }
.app-tabs { margin-top: 4px; }
.pro-card {
    padding: 22px !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-sm) !important;
}
.card-heading { margin-bottom: 16px; }
.card-heading .overline { color: var(--blue-700); font-size: 10px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.card-heading h2 { margin: 4px 0 3px !important; color: var(--ink); font-size: 21px !important; letter-spacing: -.02em; }
.card-heading p { margin: 0 !important; color: var(--muted); font-size: 12px; line-height: 1.5; }
.model-dossier { margin: 10px 0 4px; overflow: hidden; border: 1px solid var(--line); border-radius: 13px; background: #f9fbfd; }
.dossier-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 15px; border-bottom: 1px solid var(--line); background: white; }
.dossier-title { color: var(--ink); font-size: 15px; font-weight: 800; }
.dossier-badge { padding: 5px 8px; border-radius: 999px; color: var(--success); background: var(--success-bg); font-size: 10px; font-weight: 800; }
.dossier-grid { display: grid; grid-template-columns: 1fr 1fr; }
.dossier-item { padding: 12px 14px; border-bottom: 1px solid var(--line); }
.dossier-item:nth-child(odd) { border-right: 1px solid var(--line); }
.dossier-item label { display: block; margin-bottom: 3px; color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.dossier-item span { color: var(--body); font-size: 11px; line-height: 1.45; }
.dossier-caveat { padding: 11px 14px; color: #6d5420; background: var(--warning-bg); font-size: 10px; line-height: 1.45; }
.field-note { margin: -2px 0 7px; padding: 9px 11px; color: var(--body); background: #f7f9fc; border-left: 3px solid var(--line-strong); border-radius: 7px; font-size: 10px; line-height: 1.45; }
.button-row { gap: 8px !important; margin-top: 8px; }
#generate-btn { min-height: 50px; font-weight: 750; }
#generate-btn button { min-height: 50px; color: white !important; background: var(--blue-700) !important; border: 1px solid var(--blue-700) !important; box-shadow: 0 8px 20px rgba(23,92,211,.18); }
#generate-btn button:hover { background: #124eae !important; }
#clear-btn button { min-height: 50px; color: var(--body) !important; background: white !important; border: 1px solid var(--line-strong) !important; }
.run-note { min-height: 42px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 9px; color: var(--body); background: #f8fafc; font-size: 11px; line-height: 1.45; }
.results-card { min-height: 670px; }
.results-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.results-header h2 { margin: 0 !important; color: var(--ink); font-size: 21px !important; }
.results-header span { padding: 6px 9px; border-radius: 999px; color: var(--muted); background: #f1f4f8; font-size: 10px; font-weight: 800; }
.empty-state { display: grid; place-items: center; min-height: 172px; padding: 24px; text-align: center; border: 1px dashed var(--line-strong); border-radius: 13px; background: #fbfcfe; }
.empty-icon { display: grid; place-items: center; width: 48px; height: 48px; margin: 0 auto 10px; border-radius: 14px; color: var(--blue-700); background: var(--blue-50); font-size: 22px; }
.empty-state strong { display: block; color: var(--ink); font-size: 14px; }
.empty-state p { max-width: 390px; margin: 5px auto 0 !important; color: var(--muted); font-size: 11px; line-height: 1.5; }
.result-summary { padding: 15px; border: 1px solid #bde4cd; border-radius: 13px; background: var(--success-bg); }
.result-summary-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.result-summary h3 { margin: 0 !important; color: #0e6038; font-size: 15px !important; }
.result-summary p { margin: 4px 0 0 !important; color: #35775a; font-size: 10px; }
.summary-time { color: var(--success); font-size: 11px; font-weight: 800; }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 11px; }
.metric-card { padding: 11px 12px; border: 1px solid rgba(20,128,74,.16); border-radius: 9px; background: rgba(255,255,255,.72); }
.metric-card label { display: block; color: #4d7b64; font-size: 9px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
.metric-card strong { color: #123f2a; font-size: 18px; }
.metric-card span { display: block; color: #5f806d; font-size: 9px; }
.gallery-shell { margin-top: 12px; }
.compare-intro, .method-card { padding: 20px; border: 1px solid var(--line); border-radius: var(--radius-md); background: var(--surface); }
.compare-intro h2, .method-card h3 { margin: 0 0 6px !important; color: var(--ink); }
.compare-intro p, .method-card p { margin: 0 !important; color: var(--body); font-size: 12px; line-height: 1.6; }
.model-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }
.model-card { padding: 17px; border: 1px solid var(--line); border-radius: 14px; background: var(--surface); box-shadow: var(--shadow-sm); }
.model-card-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
.model-card h3 { margin: 0 !important; color: var(--ink); font-size: 15px !important; }
.model-family { color: var(--muted); font-size: 10px; }
.fid-chip { padding: 6px 8px; border-radius: 8px; color: var(--blue-700); background: var(--blue-50); font-size: 10px; font-weight: 800; white-space: nowrap; }
.model-card p { min-height: 38px; margin: 12px 0 !important; color: var(--body); font-size: 11px; line-height: 1.5; }
.model-meta { display: flex; justify-content: space-between; padding-top: 10px; border-top: 1px solid var(--line); color: var(--muted); font-size: 9px; }
.method-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }
.pipeline-step { position: relative; padding: 16px; border: 1px solid var(--line); border-radius: 12px; background: #f9fbfd; }
.pipeline-step strong { display: block; color: var(--ink); font-size: 12px; }
.pipeline-step span { display: block; margin-top: 4px; color: var(--muted); font-size: 10px; line-height: 1.45; }
.disclaimer { margin-top: 14px; padding: 14px 16px; color: #6d5420; background: var(--warning-bg); border: 1px solid #f3dfa3; border-radius: 12px; font-size: 11px; line-height: 1.55; }
.app-footer { display: flex; justify-content: space-between; gap: 16px; margin-top: 26px; padding: 20px 2px; color: var(--muted); border-top: 1px solid var(--line); font-size: 10px; }
:focus-visible { outline: 3px solid rgba(36,116,229,.35) !important; outline-offset: 2px !important; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; } }
@media (max-width: 1040px) {
    .product-hero { grid-template-columns: 1fr; padding: 34px; }
    .scan-visual { display: none; }
    .model-grid, .method-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 780px) {
    .gradio-container { padding: 0 12px 30px !important; }
    .app-header { margin: 0 -12px; padding: 0 15px; }
    .header-badges .header-badge:first-child { display: none; }
    .product-hero { margin-top: 14px; padding: 26px 22px; border-radius: 16px; }
    .product-hero h1 { font-size: 38px !important; }
    .workflow-strip, .model-grid, .method-grid, .pipeline { grid-template-columns: 1fr; }
    .workflow-step + .workflow-step { border-left: 0; border-top: 1px solid var(--line); }
    .pro-card { padding: 16px !important; }
    .results-card { min-height: auto; }
    .app-footer { flex-direction: column; }
}
@media (forced-colors: active) { .hero-chip, .header-badge, .fid-chip { border: 1px solid CanvasText; } }
"""


# =============================================================================
# Model Definitions
# =============================================================================

class DCGANGenerator(nn.Module):
    def __init__(self, noise_dim=100, channels=1, feature_g=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class WGANGPGenerator(nn.Module):
    def __init__(self, noise_dim=100, channels=1, feature_g=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# VQ-VAE components
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embed_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embed_dim)
        d = (z_flat ** 2).sum(dim=1, keepdim=True) + \
            (self.embedding.weight ** 2).sum(dim=1) - \
            2 * z_flat @ self.embedding.weight.t()
        indices = d.argmin(dim=1)
        z_q = self.embedding(indices).view(z.shape)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, codebook_loss, commitment_loss, indices


class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)


class VQVAEDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, embed_dim=64, num_embeddings=512, beta=0.25):
        super().__init__()
        self.encoder = VQVAEEncoder(in_channels, hidden_dim, embed_dim)
        self.vq = VectorQuantizer(num_embeddings, embed_dim, beta)
        self.decoder = VQVAEDecoder(embed_dim, hidden_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, codebook_loss, commitment_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, codebook_loss, commitment_loss, indices

    def decode(self, z_q):
        return self.decoder(z_q)


# Flow Matching v2 components
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return out + residual


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.enc1 = ResBlock(in_ch, base_ch, time_emb_dim)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_emb_dim)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        self.enc4 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1)
        self.attn_enc3 = SelfAttention(base_ch * 4, num_heads=4)
        self.attn_enc4 = SelfAttention(base_ch * 4, num_heads=4)
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.mid_attn = SelfAttention(base_ch * 4, num_heads=4)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.up4 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, 2, 1)
        self.dec4 = ResBlock(base_ch * 8, base_ch * 4, time_emb_dim)
        self.attn_dec4 = SelfAttention(base_ch * 4, num_heads=4)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.dec3 = ResBlock(base_ch * 4, base_ch * 2, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.dec2 = ResBlock(base_ch * 2, base_ch, time_emb_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act = nn.SiLU()
        self.out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        e3 = self.attn_enc3(e3)
        e4 = self.enc4(self.down3(e3), t_emb)
        e4 = self.attn_enc4(e4)
        m = self.mid1(e4, t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)
        d4 = self.dec4(torch.cat([self.up4(m), e3], dim=1), t_emb)
        d4 = self.attn_dec4(d4)
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1), t_emb)
        return self.out(self.out_act(self.out_norm(d2)))


class ODEFunc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        t_batch = t.expand(x.shape[0])
        return self.model(x, t_batch)


# =============================================================================
# Model Registry
# =============================================================================

MODEL_CONFIGS = {
    "flow_matching_v2": {
        "name": "Flow Matching v2",
        "emoji": "🥇",
        "fid": 6.20,
        "description": "Best model — Attention + EMA + 100 epochs",
        "checkpoint": OUTPUTS_DIR / "06_flow_matching_v2" / "checkpoints" / "checkpoint_epoch_100.pt",
        "type": "flow_matching_v2"
    },
    "ddpm": {
        "name": "DDPM",
        "emoji": "🥈",
        "fid": 8.96,
        "description": "Diffusion model baseline",
        "checkpoint": OUTPUTS_DIR / "05_ddpm" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "ddpm"
    },
    "wgan_gp": {
        "name": "WGAN-GP",
        "emoji": "🥉",
        "fid": 11.10,
        "description": "Stable GAN with gradient penalty",
        "checkpoint": OUTPUTS_DIR / "03_wgan_gp" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "wgan_gp"
    },
    "dcgan": {
        "name": "DCGAN",
        "emoji": "4️⃣",
        "fid": 15.24,
        "description": "Classic GAN baseline",
        "checkpoint": OUTPUTS_DIR / "02_dcgan" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "dcgan"
    },
    "vqvae": {
        "name": "VQ-VAE",
        "emoji": "5️⃣",
        "fid": 46.59,
        "description": "Discrete codebook (no prior)",
        "checkpoint": OUTPUTS_DIR / "04_vqvae" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "vqvae"
    },
    "stable_diffusion": {
        "name": "Stable Diffusion",
        "emoji": "6️⃣",
        "fid": 94.71,
        "description": "img2img mode (different input)",
        "checkpoint": None,
        "check_path": OUTPUTS_DIR / "01_stable_diffusion" / "metrics.json",
        "type": "stable_diffusion"
    }
}

loaded_models = {}
WEIGHTS_REPO_ID = os.getenv(
    "WEIGHTS_REPO_ID",
    "lakshyalol/fake-dataset-factory-weights",
)


def resolve_checkpoint(config: dict):
    """Prefer unchanged local weights; download only the selected missing model."""
    checkpoint_path = config.get("checkpoint")
    if checkpoint_path is None or checkpoint_path.exists():
        return checkpoint_path

    from huggingface_hub import hf_hub_download

    filename = checkpoint_path.relative_to(PROJECT_ROOT).as_posix()
    try:
        return Path(
            hf_hub_download(
                repo_id=WEIGHTS_REPO_ID,
                filename=filename,
                repo_type="model",
                local_dir=PROJECT_ROOT,
            )
        )
    except Exception as exc:
        print(f"Could not download {filename}: {exc}")
        return None


def check_model_available(model_id: str) -> bool:
    config = MODEL_CONFIGS[model_id]
    if config["type"] == "stable_diffusion":
        check_path = config.get("check_path")
        return bool(check_path and check_path.exists() and DIFFUSERS_AVAILABLE)
    checkpoint = config.get("checkpoint")
    return bool((checkpoint and checkpoint.exists()) or WEIGHTS_REPO_ID)


def get_available_models() -> dict:
    return {mid: check_model_available(mid) for mid in MODEL_CONFIGS}


def load_model(model_id: str):
    if model_id in loaded_models:
        return loaded_models[model_id]

    if not check_model_available(model_id):
        return None

    config = MODEL_CONFIGS[model_id]
    model_type = config["type"]

    if model_type == "stable_diffusion":
        if not DIFFUSERS_AVAILABLE:
            return None
        loaded_models[model_id] = "stable_diffusion"
        return "stable_diffusion"

    checkpoint_path = resolve_checkpoint(config)
    if checkpoint_path is None:
        return None

    if model_type == "dcgan":
        model = DCGANGenerator(NOISE_DIM, 1, FEATURE_G)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["netG_state_dict"])
        model.to(device).eval()
        loaded_models[model_id] = model
        return model

    elif model_type == "wgan_gp":
        model = WGANGPGenerator(NOISE_DIM, 1, FEATURE_G)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["netG_state_dict"])
        model.to(device).eval()
        loaded_models[model_id] = model
        return model

    elif model_type == "vqvae":
        model = VQVAE(in_channels=1, hidden_dim=128, embed_dim=64, num_embeddings=512, beta=0.25)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        loaded_models[model_id] = model
        return model

    elif model_type == "ddpm":
        if not DIFFUSERS_AVAILABLE:
            return None
        model = UNet2DModel(
            sample_size=IMG_SIZE, in_channels=1, out_channels=1, layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        loaded_models[model_id] = {"model": model, "scheduler": scheduler}
        return loaded_models[model_id]

    elif model_type == "flow_matching_v2":
        if not TORCHDIFFEQ_AVAILABLE:
            return None
        model = AttentionUNet(in_ch=1, base_ch=64, time_emb_dim=256)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # load EMA weights if available
        if "ema_shadow" in checkpoint:
            for name, param in model.named_parameters():
                if name in checkpoint["ema_shadow"]:
                    param.data = checkpoint["ema_shadow"][name]
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        loaded_models[model_id] = model
        return model

    return None


# =============================================================================
# Evaluation
# =============================================================================

def load_xrv_model():
    if not XRV_AVAILABLE:
        return None, None
    if "_xrv" in loaded_models:
        return loaded_models["_xrv"]

    xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    xrv_model.to(device).eval()
    feature_extractor = nn.Sequential(*list(xrv_model.features.children()))
    feature_extractor.to(device).eval()
    loaded_models["_xrv"] = (xrv_model, feature_extractor)
    return loaded_models["_xrv"]


def preprocess_for_xrv(img: Image.Image) -> torch.Tensor:
    img = img.convert('L').resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 255.0) * 2048 - 1024
    return torch.tensor(arr[np.newaxis, ...], dtype=torch.float32)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    if len(real_features) < 2 or len(fake_features) < 2:
        return None
    mu_real, mu_fake = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
    sigma_real, sigma_fake = np.cov(real_features, rowvar=False), np.cov(fake_features, rowvar=False)
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps
    diff = mu_real - mu_fake
    try:
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        if not np.isfinite(covmean).all():
            return None
        fid = float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))
        return fid if np.isfinite(fid) else None
    except Exception:
        return None


# =============================================================================
# Generation Functions
# =============================================================================

def generate_gan_images(model, n_samples: int) -> list:
    with torch.inference_mode():
        noise = torch.randn(n_samples, NOISE_DIM, 1, 1, device=device)
        batch = ((model(noise).squeeze(1).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(arr, mode='L') for arr in batch]


def generate_sd_images(n_samples: int, target_class: str, inference_steps: int) -> list:
    if not DIFFUSERS_AVAILABLE:
        return []
    class_dir = DATA_DIR / target_class.lower()
    if not class_dir.exists():
        return []
    real_paths = sorted([p for p in class_dir.iterdir() if p.suffix == '.png'])[:n_samples]
    if not real_paths:
        return []

    pipe = loaded_models.get("_sd_pipe")
    if pipe is None:
        inference_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=inference_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        pipe.enable_attention_slicing()
        loaded_models["_sd_pipe"] = pipe

    images = []
    prompt = "chest x-ray, medical radiograph, grayscale"
    neg_prompt = "color, artistic, painting, drawing, cartoon"
    generator_device = "cuda" if device.type == "cuda" else "cpu"

    for i, img_path in enumerate(real_paths):
        img = Image.open(img_path).convert('RGB').resize((512, 512), Image.LANCZOS)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=img,
                strength=0.7,
                guidance_scale=7.5,
                num_inference_steps=inference_steps,
                generator=torch.Generator(device=generator_device).manual_seed(SEED + i),
            )
        images.append(result.images[0].convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))

    return images


def generate_vqvae_images(model, n_samples: int) -> list:
    with torch.inference_mode():
        indices = torch.randint(0, 512, (n_samples, 16 * 16), device=device)
        z_q = model.vq.embedding(indices).view(n_samples, 16, 16, 64).permute(0, 3, 1, 2).contiguous()
        batch = (model.decode(z_q).squeeze(1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(arr, mode='L') for arr in batch]


def generate_ddpm_images(model_dict, n_samples: int, inference_steps: int) -> list:
    model, scheduler = model_dict["model"], model_dict["scheduler"]
    scheduler.set_timesteps(inference_steps, device=device)
    images = []
    batch_limit = 2 if device.type == "cpu" else 4

    with torch.inference_mode():
        for offset in range(0, n_samples, batch_limit):
            batch_size = min(batch_limit, n_samples - offset)
            torch.manual_seed(SEED + offset)
            sample = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
            for timestep in scheduler.timesteps:
                prediction = model(sample, timestep).sample
                sample = scheduler.step(prediction, timestep, sample).prev_sample

            batch = ((sample.squeeze(1).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            images.extend(Image.fromarray(arr, mode='L') for arr in batch)
    return images


def generate_flow_matching_images(model, n_samples: int, inference_steps: int) -> list:
    images = []
    batch_limit = 2 if device.type == "cpu" else 4
    dt = 1.0 / inference_steps

    with torch.inference_mode():
        for offset in range(0, n_samples, batch_limit):
            batch_size = min(batch_limit, n_samples - offset)
            torch.manual_seed(SEED + offset)
            sample = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)

            # Fixed-step Euler integration is predictable and dramatically
            # faster than an adaptive solver for an interactive demo.
            for step in range(inference_steps):
                time = torch.full((batch_size,), step * dt, device=device)
                sample = sample + dt * model(sample, time)

            batch = ((sample.squeeze(1).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            images.extend(Image.fromarray(arr, mode='L') for arr in batch)
    return images


QUALITY_PRESETS = {
    "Fast": {"ddpm": 25, "flow_matching_v2": 10, "stable_diffusion": 15},
    "Balanced": {"ddpm": 50, "flow_matching_v2": 20, "stable_diffusion": 25},
    "Quality": {"ddpm": 100, "flow_matching_v2": 40, "stable_diffusion": 40},
}


def evaluate_images(images: list, target_class: str):
    xrv_model, feature_extractor = load_xrv_model()
    if xrv_model is None:
        return None, None

    batch_size = 4 if device.type == "cpu" else 8
    fake_inputs = torch.stack([preprocess_for_xrv(img) for img in images])
    fake_features, pneumonia_scores = [], []
    pneumonia_idx = next(
        (i for i, name in enumerate(xrv_model.pathologies) if 'lung opacity' in name.lower()),
        None,
    )

    with torch.inference_mode():
        for offset in range(0, len(fake_inputs), batch_size):
            inputs = fake_inputs[offset:offset + batch_size].to(device)
            fake_features.append(feature_extractor(inputs).mean(dim=[2, 3]).cpu().numpy())
            if pneumonia_idx is not None:
                pneumonia_scores.extend(xrv_model(inputs)[:, pneumonia_idx].cpu().tolist())

    fid_score = None
    class_dir = DATA_DIR / target_class.lower()
    real_paths = sorted(class_dir.glob('*.png')) if class_dir.exists() else []
    if real_paths and len(images) >= 2:
        selected = random.sample(real_paths, min(len(images), len(real_paths)))
        real_inputs = torch.stack([preprocess_for_xrv(Image.open(path)) for path in selected])
        real_features = []
        with torch.inference_mode():
            for offset in range(0, len(real_inputs), batch_size):
                inputs = real_inputs[offset:offset + batch_size].to(device)
                real_features.append(feature_extractor(inputs).mean(dim=[2, 3]).cpu().numpy())
        fid_score = compute_fid(
            np.concatenate(real_features, axis=0),
            np.concatenate(fake_features, axis=0),
        )

    tstr_accuracy = None
    if pneumonia_scores:
        positive = target_class.lower() == "pneumonia"
        correct = sum((score > 0.5) == positive for score in pneumonia_scores)
        tstr_accuracy = correct / len(pneumonia_scores) * 100
    return fid_score, tstr_accuracy


def generate_images(
    model_id: str,
    target_class: str,
    n_samples: int,
    quality: str = "Fast",
    run_evaluation: bool = False,
    progress=gr.Progress(),
):
    config = MODEL_CONFIGS[model_id]
    progress(0.02, desc=f"Loading {config['name']}...")
    model = load_model(model_id)
    if model is None:
        return None, None, None, None, "Model could not be loaded"

    model_type = config["type"]
    steps = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Fast"]).get(model_type)
    progress(0.12, desc=f"Generating with {config['name']}...")

    if model_type == "stable_diffusion":
        images = generate_sd_images(n_samples, target_class, steps)
    elif model_type in ("dcgan", "wgan_gp"):
        images = generate_gan_images(model, n_samples)
    elif model_type == "vqvae":
        images = generate_vqvae_images(model, n_samples)
    elif model_type == "ddpm":
        images = generate_ddpm_images(model, n_samples, steps)
    elif model_type == "flow_matching_v2":
        images = generate_flow_matching_images(model, n_samples, steps)
    else:
        return None, None, None, None, "Unknown model type"

    if not images:
        return None, None, None, None, "Generation failed"

    fid_score, tstr_accuracy = config["fid"], None
    if run_evaluation:
        progress(0.78, desc="Running optional X-ray evaluation...")
        measured_fid, tstr_accuracy = evaluate_images(images, target_class)
        if measured_fid is not None:
            fid_score = measured_fid

    progress(0.92, desc="Packaging images...")
    tmp_dir = tempfile.mkdtemp()
    zip_dir = Path(tmp_dir) / "generated"
    zip_dir.mkdir()
    for index, image in enumerate(images):
        image.save(zip_dir / f"{index:04d}.png", "PNG")

    zip_path = Path(tmp_dir) / f"{model_id}_{target_class}_generated.zip"
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', zip_dir)

    step_note = f" · {steps} steps" if steps else ""
    metric_note = " · evaluated" if run_evaluation else " · benchmark FID"
    status = f"Generated {len(images)} images with {config['name']}{step_note}{metric_note}"
    return images, fid_score, tstr_accuracy, str(zip_path), status


# =============================================================================
# Gradio Interface
# =============================================================================

def create_app():
    available = get_available_models()
    n_available = sum(available.values())
    device_label = (
        "Apple GPU · Metal" if device.type == "mps"
        else "NVIDIA GPU · CUDA" if device.type == "cuda"
        else "Hosted CPU" if os.getenv("SPACE_ID") else "Local CPU"
    )
    default_model = "dcgan" if available.get("dcgan") else next(
        (model_id for model_id, ready in available.items() if ready),
        None,
    )

    model_guides = {
        "flow_matching_v2": {
            "label": "Flow Matching v2 — strongest benchmark",
            "family": "Continuous flow · 2022",
            "best": "Highest benchmark quality",
            "how": "Moves noise to an image through a learned velocity field.",
            "conditioning": "Not runtime-conditioned",
            "preset": "10 / 20 / 40 ODE steps",
            "runtime": "Moderate",
            "caveat": "The interactive app uses fixed-step integration for speed; the saved benchmark used the project evaluation setup.",
        },
        "ddpm": {
            "label": "DDPM — diffusion baseline",
            "family": "Diffusion · 2020",
            "best": "High-fidelity diffusion samples",
            "how": "Iteratively removes noise with a trained UNet.",
            "conditioning": "Not runtime-conditioned",
            "preset": "25 / 50 / 100 DDIM steps",
            "runtime": "Moderate",
            "caveat": "More inference steps can improve refinement but increase waiting time almost linearly.",
        },
        "wgan_gp": {
            "label": "WGAN-GP — stable fast GAN",
            "family": "Generative adversarial network · 2017",
            "best": "Fast batches with stable GAN training",
            "how": "Transforms a random latent vector in one generator pass.",
            "conditioning": "Not runtime-conditioned",
            "preset": "Single pass",
            "runtime": "Fast",
            "caveat": "The reference cohort does not steer this checkpoint at generation time.",
        },
        "dcgan": {
            "label": "DCGAN — recommended first run",
            "family": "Generative adversarial network · 2014",
            "best": "Fastest introduction to the app",
            "how": "Transforms a random latent vector in one generator pass.",
            "conditioning": "Not runtime-conditioned",
            "preset": "Single pass",
            "runtime": "Fastest",
            "caveat": "Use this model to verify your setup before testing iterative models.",
        },
        "vqvae": {
            "label": "VQ-VAE — discrete codebook",
            "family": "Latent autoencoder · 2017",
            "best": "Exploring discrete latent representations",
            "how": "Decodes randomly sampled vectors from a learned codebook.",
            "conditioning": "Not runtime-conditioned",
            "preset": "Single pass",
            "runtime": "Fast",
            "caveat": "No learned PixelCNN prior is included, so random codebook sampling limits generation quality.",
        },
        "stable_diffusion": {
            "label": "Stable Diffusion — X-ray variation",
            "family": "Latent diffusion img2img · 2022",
            "best": "Creating variations from source X-rays",
            "how": "Adds noise to a real source X-ray and denoises a variation.",
            "conditioning": "Source cohort affects input pool",
            "preset": "15 / 25 / 40 steps",
            "runtime": "Slow first run",
            "caveat": "This is img2img, not generation from pure noise; its benchmark FID is not directly comparable.",
        },
    }

    def model_dossier(model_id):
        if not model_id:
            return '<div class="model-dossier"><div class="dossier-caveat">Select a model to continue.</div></div>'
        guide = model_guides[model_id]
        config = MODEL_CONFIGS[model_id]
        readiness = "Ready" if available.get(model_id) else "Weights unavailable"
        return f"""
        <section class="model-dossier">
            <div class="dossier-head">
                <div><div class="dossier-title">{config['name']}</div><div class="model-family">{guide['family']}</div></div>
                <div class="dossier-badge">{readiness}</div>
            </div>
            <div class="dossier-grid">
                <div class="dossier-item"><label>Best for</label><span>{guide['best']}</span></div>
                <div class="dossier-item"><label>Expected runtime</label><span>{guide['runtime']}</span></div>
                <div class="dossier-item"><label>How it generates</label><span>{guide['how']}</span></div>
                <div class="dossier-item"><label>Benchmark</label><span>FID {config['fid']:.2f} ↓ · saved 100-image run</span></div>
                <div class="dossier-item"><label>Conditioning</label><span>{guide['conditioning']}</span></div>
                <div class="dossier-item"><label>Preset behavior</label><span>{guide['preset']}</span></div>
            </div>
            <div class="dossier-caveat"><strong>Know before running:</strong> {guide['caveat']}</div>
        </section>
        """

    def cohort_note(model_id):
        if model_id == "stable_diffusion":
            return '<div class="field-note"><strong>Source cohort:</strong> selects the real X-ray pool used as img2img input. It changes the source images supplied to Stable Diffusion.</div>'
        return '<div class="field-note"><strong>Reference only:</strong> this choice is used for optional evaluation and run context. It does not condition or medically label pixels generated by this checkpoint.</div>'

    def empty_results():
        return """
        <section class="empty-state">
            <div><div class="empty-icon">✦</div><strong>No generated batch yet</strong>
            <p>Start with DCGAN, 4 images, and Fast mode. Your images, run summary, and ZIP export will appear here.</p></div>
        </section>
        """

    def ready_note(model_id):
        if not model_id:
            return '<div class="run-note">Choose a model to begin.</div>'
        guide = model_guides[model_id]
        return f'<div class="run-note"><strong>Ready.</strong> {guide["label"]} · {guide["runtime"]} runtime. First use may download its checkpoint.</div>'

    def model_controls(model_id):
        iterative = model_id in {"flow_matching_v2", "ddpm", "stable_diffusion"}
        max_images = 2 if model_id == "stable_diffusion" else 4 if iterative else 16
        count = 1 if iterative else 4
        cohort_label = "Source cohort" if model_id == "stable_diffusion" else "Reference cohort"
        return (
            model_dossier(model_id),
            cohort_note(model_id),
            gr.update(interactive=iterative, value="Fast"),
            gr.update(maximum=max_images, value=count),
            gr.update(label=cohort_label),
            ready_note(model_id),
        )

    model_cards = "".join(
        f"""
        <article class="model-card">
            <div class="model-card-top"><div><h3>{MODEL_CONFIGS[model_id]['name']}</h3><div class="model-family">{guide['family']}</div></div><div class="fid-chip">FID {MODEL_CONFIGS[model_id]['fid']:.2f}</div></div>
            <p>{guide['best']}. {guide['how']}</p>
            <div class="model-meta"><span>{guide['runtime']}</span><span>{guide['conditioning']}</span></div>
        </article>
        """
        for model_id, guide in model_guides.items()
    )

    hero_html = f"""
    <header class="app-header">
        <div class="brand"><div class="brand-mark">F</div><div class="brand-copy"><strong>Fake Dataset Factory</strong><span>Synthetic imaging research tool</span></div></div>
        <div class="header-badges"><div class="header-badge">{n_available} models ready</div><div class="header-badge live">{device_label}</div></div>
    </header>
    <section class="product-hero">
        <div>
            <div class="hero-kicker">Interactive generative model laboratory</div>
            <h1>Build synthetic X-ray datasets, one model at a time.</h1>
            <p>Explore six generative architectures, understand how each one works, generate controlled-size batches, and export the results for downstream research.</p>
            <div class="hero-actions"><span class="hero-chip">6 generation engines</span><span class="hero-chip">64×64 grayscale PNG</span><span class="hero-chip caution">Research use only · not for diagnosis</span></div>
        </div>
        <div class="scan-visual" aria-hidden="true">
            <div class="scan-frame"><div class="scan-topline"><span>SYNTHETIC STUDY</span><span>64 × 64</span></div>
                <svg viewBox="0 0 280 220" role="img">
                    <defs><radialGradient id="g" cx="50%" cy="45%" r="65%"><stop offset="0" stop-color="#426480"/><stop offset="1" stop-color="#0d2135"/></radialGradient></defs>
                    <rect width="280" height="220" fill="url(#g)"/><path d="M140 26v168" stroke="#9ab2c6" stroke-width="7" opacity=".36"/>
                    <path d="M126 48C91 49 58 75 53 122c-4 39 17 70 52 70 21 0 29-18 29-39V66c0-11-3-18-8-18Z" fill="#b8c9d7" opacity=".58"/>
                    <path d="M154 48c35 1 68 27 73 74 4 39-17 70-52 70-21 0-29-18-29-39V66c0-11 3-18 8-18Z" fill="#b8c9d7" opacity=".58"/>
                    <g fill="none" stroke="#dbe6ee" opacity=".25"><path d="M34 65q106 44 212 0"/><path d="M27 88q113 45 226 0"/><path d="M23 112q117 42 234 0"/><path d="M24 137q116 38 232 0"/><path d="M31 162q109 32 218 0"/></g>
                    <circle cx="140" cy="109" r="6" fill="#53d5e3"/><path d="M140 91v36M122 109h36" stroke="#53d5e3" opacity=".75"/>
                </svg>
            </div><div class="scan-tag">Synthetic preview</div>
        </div>
    </section>
    <section class="workflow-strip">
        <div class="workflow-step"><div class="step-number">1</div><div><strong>Choose an engine</strong><span>Compare model behavior, speed, and caveats.</span></div></div>
        <div class="workflow-step"><div class="step-number">2</div><div><strong>Configure a small batch</strong><span>Pick the reference cohort, size, and quality preset.</span></div></div>
        <div class="workflow-step"><div class="step-number">3</div><div><strong>Review and export</strong><span>Inspect the gallery and download PNGs as a ZIP.</span></div></div>
    </section>
    """

    theme = gr.themes.Base(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate")
    with gr.Blocks(title="Fake Dataset Factory", css=CUSTOM_CSS, theme=theme) as app:
        gr.HTML(hero_html)
        with gr.Tabs(elem_classes="app-tabs"):
            with gr.Tab("Generate", id="generate-tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=350, elem_classes="pro-card"):
                        gr.HTML('<div class="card-heading"><div class="overline">Step 1 · Configure</div><h2>Design your batch</h2><p>Start with the recommended setup, then explore model trade-offs.</p></div>')
                        model_dropdown = gr.Dropdown(
                            choices=[(guide["label"], model_id) for model_id, guide in model_guides.items() if available.get(model_id)],
                            value=default_model,
                            label="Generation engine",
                            info="Each engine uses a different generation strategy.",
                        )
                        model_info = gr.HTML(model_dossier(default_model))
                        class_dropdown = gr.Radio(choices=["Pneumonia", "Normal"], value="Pneumonia", label="Reference cohort")
                        class_help = gr.HTML(cohort_note(default_model))
                        n_samples = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch size", info="Iterative models automatically use safer limits.")
                        quality = gr.Radio(choices=["Fast", "Balanced", "Quality"], value="Fast", label="Speed / quality", interactive=False, info="Enabled for iterative models only.")
                        with gr.Accordion("Advanced evaluation", open=False):
                            run_evaluation = gr.Checkbox(value=False, label="Evaluate this batch (slower)", info="Loads a separate DenseNet model for proxy TSTR and exploratory FID. Batch FID needs at least 2 images.")
                        run_note = gr.HTML(ready_note(default_model))
                        with gr.Row(elem_classes="button-row"):
                            generate_btn = gr.Button("Generate batch", variant="primary", size="lg", scale=3, elem_id="generate-btn")
                            reset_btn = gr.Button("Reset", size="lg", scale=1, elem_id="clear-btn")

                    with gr.Column(scale=7, min_width=520, elem_classes=["pro-card", "results-card"]):
                        gr.HTML('<div class="results-header"><h2>Batch results</h2><span>PNG · 64×64</span></div>')
                        result_summary = gr.HTML(empty_results())
                        gallery = gr.Gallery(label="Synthetic X-ray gallery", columns=4, rows=4, height=430, object_fit="contain", show_label=True, visible=False, elem_classes="gallery-shell")
                        with gr.Row():
                            download_btn = gr.File(label="Download batch as ZIP", visible=False, scale=3)
                            clear_btn = gr.Button("Clear results", scale=1)

            with gr.Tab("Compare models", id="compare-tab"):
                gr.HTML(f"""
                <section class="compare-intro"><h2>Choose the right generation engine</h2><p><strong>FID</strong> measures feature-distribution distance; lower is better. Values below are saved benchmarks from 100-image project runs—not scores computed for the current batch. Stable Diffusion uses img2img and is not directly comparable to pure-noise generators.</p></section>
                <section class="model-grid">{model_cards}</section>
                <div class="disclaimer"><strong>Conditioning note:</strong> the custom checkpoints in this app are not runtime-conditioned by the Normal/Pneumonia selector. That choice provides an evaluation reference. Stable Diffusion uses it to choose the source-image cohort.</div>
                """)

            with gr.Tab("How it works", id="method-tab"):
                gr.HTML("""
                <section class="method-grid">
                    <article class="method-card"><h3>What this tool does</h3><p>Runs trained generative models in inference mode to produce synthetic grayscale chest X-rays. It does not train models in the browser and does not make clinical predictions.</p></article>
                    <article class="method-card"><h3>Domain-adapted FID</h3><p>Uses TorchXRayVision DenseNet features instead of ImageNet features. Saved benchmarks use 100 images; tiny live batches are exploratory and statistically weak.</p></article>
                    <article class="method-card"><h3>Proxy TSTR</h3><p>Reports agreement with a pretrained Lung Opacity classifier. It is a research proxy—not diagnostic accuracy and not evidence of clinical validity.</p></article>
                </section>
                <section class="compare-intro" style="margin-top:12px"><h2>Generation pipeline</h2><div class="pipeline">
                    <div class="pipeline-step"><strong>1 · Select</strong><span>Choose an architecture based on speed, quality, and generation method.</span></div>
                    <div class="pipeline-step"><strong>2 · Load</strong><span>The checkpoint loads locally or downloads once from the public model repository.</span></div>
                    <div class="pipeline-step"><strong>3 · Generate</strong><span>The model transforms noise—or a source image for img2img—into synthetic pixels.</span></div>
                    <div class="pipeline-step"><strong>4 · Export</strong><span>Review the output grid and download the generated PNG batch as a ZIP.</span></div>
                </div></section>
                <div class="disclaimer"><strong>Responsible use:</strong> outputs may contain artifacts, memorized patterns, or medically implausible structures. Do not use them for diagnosis, treatment, or unsupervised clinical decision-making.</div>
                """)

        gr.HTML('<footer class="app-footer"><span>Fake Dataset Factory · CSET419 Generative AI Project</span><span>Research demo · Synthetic outputs are not medical diagnoses</span></footer>')

        def on_generate(model_id, target_class, count, preset, evaluate):
            if not model_id:
                gr.Error("Select a generation engine first.")
                return gr.update(), empty_results(), gr.update(), '<div class="run-note">Select a model to continue.</div>'
            started = time.perf_counter()
            try:
                images, fid, tstr, zip_path, status = generate_images(model_id, target_class, int(count), preset, evaluate)
            except Exception as exc:
                gr.Error(f"Generation failed: {exc}")
                return gr.update(), f'<div class="dossier-caveat"><strong>Generation failed.</strong> {exc}</div>', gr.update(), ready_note(model_id)

            elapsed = time.perf_counter() - started
            config = MODEL_CONFIGS[model_id]
            metric_source = "This batch" if evaluate and int(count) >= 2 else "Saved benchmark"
            tstr_value = f"{tstr:.1f}%" if tstr is not None else "Not run"
            captions = [(image, f"Synthetic X-ray {index + 1}") for index, image in enumerate(images or [])]
            summary = f"""
            <section class="result-summary">
                <div class="result-summary-head"><div><h3>{len(images or [])} images ready</h3><p>{config['name']} · {target_class} reference · {preset} preset · 64×64 PNG</p></div><div class="summary-time">{elapsed:.1f}s</div></div>
                <div class="metric-grid">
                    <div class="metric-card"><label>FID · {metric_source}</label><strong>{fid:.2f}</strong><span>Lower is better</span></div>
                    <div class="metric-card"><label>Proxy TSTR</label><strong>{tstr_value}</strong><span>{'Current batch' if tstr is not None else 'Enable evaluation to calculate'}</span></div>
                </div>
            </section>
            """
            note = f'<div class="run-note"><strong>Complete.</strong> {status}. Your ZIP contains {len(images or [])} PNG files.</div>'
            return gr.update(value=captions, visible=True), summary, gr.update(value=zip_path, visible=True), note

        def reset_workspace():
            return (
                default_model,
                gr.update(value="Pneumonia", label="Reference cohort"),
                gr.update(value=4, maximum=16),
                gr.update(value="Fast", interactive=False),
                False,
                model_dossier(default_model),
                cohort_note(default_model),
                ready_note(default_model),
            )

        def clear_results():
            return gr.update(value=[], visible=False), empty_results(), gr.update(value=None, visible=False)

        model_dropdown.change(
            fn=model_controls,
            inputs=model_dropdown,
            outputs=[model_info, class_help, quality, n_samples, class_dropdown, run_note],
            queue=False,
            api_name=False,
        )
        generate_btn.click(
            fn=on_generate,
            inputs=[model_dropdown, class_dropdown, n_samples, quality, run_evaluation],
            outputs=[gallery, result_summary, download_btn, run_note],
            api_name="generate",
        )
        reset_btn.click(
            fn=reset_workspace,
            outputs=[model_dropdown, class_dropdown, n_samples, quality, run_evaluation, model_info, class_help, run_note],
            queue=False,
        )
        clear_btn.click(fn=clear_results, outputs=[gallery, result_summary, download_btn], queue=False)

    app.queue(default_concurrency_limit=1, max_size=8)
    return app


# Keep the previous interface available for reference, but serve the compact UI.
_legacy_create_app = create_app
from ui import build_app


def create_app():
    return build_app(
        MODEL_CONFIGS,
        get_available_models,
        generate_images,
        device,
    )


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
