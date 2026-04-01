"""
WaveletPool  –  anti-aliasing down-sampling module for tiny object detection.

Based on:  Ning & Spratling (2023), "The Importance of Anti-Aliasing in
Tiny Object Detection" — specifically the Cohen(3,3) wavelet variant and
the consistent-order placement (Figure 2c of the paper).

This module replaces standard strided-conv / max-pool down-sampling:
    Input  (B, C, H, W)  →  WaveletPool  →  Output (B, C, H/2, W/2)

It uses the 2-D Discrete Wavelet Transform (DWT) to decompose each channel
into LL, LH, HL, HH sub-bands and then reconstructs only from the LL sub-band
via the Inverse DWT.  This discards high-frequency aliasing artefacts while
preserving the low-frequency structural information needed for detection.

Usage:
    pool = WaveletPool(wavelet='coif3')
    x_down = pool(x)          # (B, C, H, W) → (B, C, H/2, W/2)
"""

import torch
import torch.nn as nn
import numpy as np

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

import config as cfg


# ─────────────────────────────────────────────────────────────────────
# Pure-PyTorch DWT / IDWT helpers (no dependency on pytorch-wavelets)
# ─────────────────────────────────────────────────────────────────────

def _get_wavelet_filters(wavelet_name: str):
    """
    Return (dec_lo, dec_hi, rec_lo, rec_hi) as 1-D numpy arrays
    for the requested wavelet.
    """
    if not HAS_PYWT:
        raise ImportError("PyWavelets (pywt) is required.  pip install PyWavelets")
    w = pywt.Wavelet(wavelet_name)
    return (np.array(w.dec_lo, dtype=np.float32),
            np.array(w.dec_hi, dtype=np.float32),
            np.array(w.rec_lo, dtype=np.float32),
            np.array(w.rec_hi, dtype=np.float32))


def _build_2d_dec_filters(dec_lo, dec_hi):
    """
    Build four 2-D decomposition kernels (LL, LH, HL, HH) from 1-D filters.
    Returns tensor of shape (4, 1, kH, kW).
    """
    lo = torch.tensor(dec_lo).unsqueeze(0)  # (1, K)
    hi = torch.tensor(dec_hi).unsqueeze(0)  # (1, K)
    # outer products
    ll = lo.T @ lo   # (K, K)
    lh = lo.T @ hi
    hl = hi.T @ lo
    hh = hi.T @ hi
    return torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # (4, 1, K, K)


def _build_2d_rec_filters(rec_lo, rec_hi):
    """
    Build four 2-D reconstruction kernels from 1-D filters.
    Returns tensor of shape (4, 1, kH, kW).
    """
    lo = torch.tensor(rec_lo).unsqueeze(0)
    hi = torch.tensor(rec_hi).unsqueeze(0)
    ll = lo.T @ lo
    lh = lo.T @ hi
    hl = hi.T @ lo
    hh = hi.T @ hi
    return torch.stack([ll, lh, hl, hh]).unsqueeze(1)


class WaveletPool(nn.Module):
    """
    Down-sample by 2x using the Discrete Wavelet Transform.

    Forward:
        x  (B, C, H, W)  →  x_down  (B, C, H/2, W/2)

    Internally performs per-channel 2-D DWT, keeps only the LL sub-band,
    then applies the 2-D IDWT on the LL sub-band (with LH=HL=HH=0)
    to produce a clean half-resolution output.
    """

    def __init__(self, wavelet: str = None):
        super().__init__()
        if wavelet is None:
            wavelet = cfg.WAVELET_NAME

        dec_lo, dec_hi, rec_lo, rec_hi = _get_wavelet_filters(wavelet)
        self.pad = len(dec_lo) // 2  # symmetric padding size

        # decomposition filters  (4, 1, K, K)
        dec = _build_2d_dec_filters(dec_lo, dec_hi)
        self.register_buffer("dec_filters", dec)

        # reconstruction filters  (4, 1, K, K)
        rec = _build_2d_rec_filters(rec_lo, rec_hi)
        self.register_buffer("rec_filters", rec)

    # ─── forward ──────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # group channels so conv2d operates per-channel
        x_flat = x.reshape(B * C, 1, H, W)

        # ── DWT (analysis) ────────────────────────────────────────
        x_pad = nn.functional.pad(x_flat, [self.pad] * 4, mode="reflect")
        coeffs = nn.functional.conv2d(x_pad, self.dec_filters, stride=2)
        # coeffs shape: (B*C, 4, H/2, W/2)

        # keep only LL sub-band; zero the rest
        ll = coeffs[:, 0:1, :, :]   # (B*C, 1, H/2, W/2)

        # ── IDWT (synthesis) from LL only ─────────────────────────
        # Up-sample LL via transposed conv with rec_LL filter
        rec_ll_filter = self.rec_filters[0:1]  # (1, 1, K, K)
        x_rec = nn.functional.conv_transpose2d(
            ll, rec_ll_filter, stride=2, padding=self.pad,
            output_padding=0,
        )

        # trim to exact half-resolution
        target_h, target_w = H // 2, W // 2
        x_rec = x_rec[:, :, :target_h, :target_w]

        return x_rec.reshape(B, C, target_h, target_w)


# ─────────────────────────────────────────────────────────────────────
# WaveletPool-aware Conv block (Conv → BN → SiLU → WaveletPool)
# Used to replace strided convolutions in the YOLO backbone.
# Key: wavelet comes AFTER convolution (consistent order, Fig 2c).
# ─────────────────────────────────────────────────────────────────────
class WaveletConv(nn.Module):
    """
    Drop-in replacement for Conv2d(stride=2):
        Conv(stride=1) → BN → SiLU → WaveletPool
    Follows the "consistent order" from the paper (filter after conv).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, wavelet=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.pool = WaveletPool(wavelet=wavelet)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


# ─────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("WaveletPool self-test ...")
    pool = WaveletPool()
    dummy = torch.randn(2, 64, 128, 128)
    out = pool(dummy)
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")
    assert out.shape == (2, 64, 64, 64), f"Unexpected shape {out.shape}"

    # gradient check
    dummy.requires_grad_(True)
    out = pool(dummy)
    loss = out.sum()
    loss.backward()
    assert dummy.grad is not None, "Gradients did not flow!"
    print("  Gradients OK ✓")

    # WaveletConv test
    wconv = WaveletConv(3, 64)
    inp = torch.randn(1, 3, 640, 640)
    y = wconv(inp)
    print(f"  WaveletConv: {inp.shape} → {y.shape}")
    print("All tests passed ✓")
