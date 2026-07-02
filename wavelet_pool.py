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
# WaveletSmooth  –  Anti-aliasing pre-filter at SAME resolution
# Inserted BEFORE existing stride-2 conv, preserving all pretrained
# weights.  Zero learnable parameters (fixed wavelet filters).
# ─────────────────────────────────────────────────────────────────────

class WaveletSmooth(nn.Module):
    """
    Anti-aliasing pre-filter at the SAME spatial resolution.

    Rather than replacing the stride-2 conv (which breaks pretrained weights),
    this module is inserted BEFORE it:
        Input (B, C, H, W) → WaveletSmooth → (B, C, H, W) → Conv(stride=2)

    Internally:  DWT → keep LL (zero LH/HL/HH) → IDWT back to original size.
    This removes high-frequency content that would alias upon downsampling,
    while keeping the existing convolution and all downstream layers intact.

    Zero learnable parameters — only fixed wavelet filter buffers.
    """

    def __init__(self, wavelet: str = None):
        super().__init__()
        if wavelet is None:
            wavelet = cfg.WAVELET_NAME

        dec_lo, dec_hi, rec_lo, rec_hi = _get_wavelet_filters(wavelet)
        self.pad_size = len(dec_lo) // 2

        dec = _build_2d_dec_filters(dec_lo, dec_hi)
        self.register_buffer("dec_filters", dec)

        rec = _build_2d_rec_filters(rec_lo, rec_hi)
        self.register_buffer("rec_filters", rec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)

        # ── DWT (analysis) ────────────────────────────────────────
        x_pad = nn.functional.pad(x_flat, [self.pad_size] * 4, mode="reflect")
        coeffs = nn.functional.conv2d(x_pad, self.dec_filters, stride=2)
        # coeffs: (B*C, 4, H/2, W/2)

        # keep only LL sub-band
        ll = coeffs[:, 0:1, :, :]

        # ── IDWT back to ORIGINAL resolution ──────────────────────
        rec_ll_filter = self.rec_filters[0:1]  # (1, 1, K, K)
        x_rec = nn.functional.conv_transpose2d(
            ll, rec_ll_filter, stride=2, padding=self.pad_size,
            output_padding=0,
        )

        # trim to original resolution (not half!)
        x_rec = x_rec[:, :, :H, :W]

        return x_rec.reshape(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────
# BlurSmooth  –  Anti-aliasing pre-filter using a fixed binomial kernel
# (BlurPool method, Zhang et al. 2019 "Making CNNs Shift-Invariant Again")
# Inserted BEFORE stride-2 conv exactly like WaveletSmooth.
# Zero learnable parameters — fixed depthwise convolution.
# ─────────────────────────────────────────────────────────────────────

class BlurSmooth(nn.Module):
    """
    Anti-aliasing pre-filter using a fixed 2-D binomial (BlurPool) kernel.

    Same interface as WaveletSmooth: input/output at SAME resolution.
    Inserted BEFORE a stride-2 conv as a low-pass pre-filter:
        Input (B, C, H, W) → BlurSmooth → (B, C, H, W) → Conv(stride=2)

    The binomial kernel (Pascal's triangle row) approximates a Gaussian
    and attenuates high-frequency content that would alias on downsampling.
    Zero learnable parameters — only a fixed depthwise-conv buffer.
    """

    def __init__(self, filter_size: int = None):
        super().__init__()
        if filter_size is None:
            filter_size = cfg.BLURPOOL_FILTER_SIZE

        kernel_1d = self._binomial_kernel(filter_size)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]   # (K, K)
        kernel_2d = kernel_2d / kernel_2d.sum()
        # (1, 1, K, K) — expanded to (C, 1, K, K) at forward time
        self.register_buffer("kernel", kernel_2d.unsqueeze(0).unsqueeze(0))
        # Asymmetric pad so spatial dims are preserved for both odd AND
        # even kernel sizes (total pad = K-1, split (K-1)//2 / rest).
        p1 = (filter_size - 1) // 2
        p2 = filter_size - 1 - p1
        self.pad = (p1, p2, p1, p2)   # (left, right, top, bottom)

    @staticmethod
    def _binomial_kernel(size: int) -> torch.Tensor:
        import math
        n = size - 1
        row = torch.tensor([math.comb(n, k) for k in range(size)], dtype=torch.float32)
        return row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kernel = self.kernel.expand(C, 1, -1, -1)   # (C, 1, K, K)
        x_pad = nn.functional.pad(x, list(self.pad), mode="reflect")
        return nn.functional.conv2d(x_pad, kernel, groups=C)


# ─────────────────────────────────────────────────────────────────────
# GaussianSmooth  –  Anti-aliasing pre-filter using a fixed Gaussian kernel
# Inserted BEFORE stride-2 conv exactly like WaveletSmooth.
# Zero learnable parameters — fixed depthwise convolution.
# ─────────────────────────────────────────────────────────────────────

class GaussianSmooth(nn.Module):
    """
    Anti-aliasing pre-filter using a fixed 2-D Gaussian kernel.

    Same interface as WaveletSmooth: input/output at SAME resolution.
    Inserted BEFORE a stride-2 conv as a low-pass pre-filter:
        Input (B, C, H, W) → GaussianSmooth → (B, C, H, W) → Conv(stride=2)

    The Gaussian kernel provides theoretically well-founded low-pass
    filtering (matched to the specified sigma) with no learnable parameters.
    """

    def __init__(self, kernel_size: int = None, sigma: float = None):
        super().__init__()
        if kernel_size is None:
            kernel_size = cfg.GAUSSIAN_KERNEL_SIZE
        if sigma is None:
            sigma = cfg.GAUSSIAN_SIGMA

        kernel_1d = self._gaussian_kernel(kernel_size, sigma)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]   # (K, K)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d.unsqueeze(0).unsqueeze(0))
        # Asymmetric pad so spatial dims are preserved for both odd AND
        # even kernel sizes (total pad = K-1, split (K-1)//2 / rest).
        p1 = (kernel_size - 1) // 2
        p2 = kernel_size - 1 - p1
        self.pad = (p1, p2, p1, p2)   # (left, right, top, bottom)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        center = size // 2
        x = torch.arange(size, dtype=torch.float32) - center
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kernel = self.kernel.expand(C, 1, -1, -1)   # (C, 1, K, K)
        x_pad = nn.functional.pad(x, list(self.pad), mode="reflect")
        return nn.functional.conv2d(x_pad, kernel, groups=C)


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

    # WaveletSmooth test (same resolution)
    smooth = WaveletSmooth()
    dummy2 = torch.randn(2, 64, 128, 128)
    out2 = smooth(dummy2)
    print(f"  WaveletSmooth: {dummy2.shape} → {out2.shape}")
    assert out2.shape == dummy2.shape, f"WaveletSmooth should preserve resolution, got {out2.shape}"
    dummy2.requires_grad_(True)
    out2 = smooth(dummy2)
    out2.sum().backward()
    assert dummy2.grad is not None, "WaveletSmooth gradients did not flow!"
    print("  WaveletSmooth OK ✓ (same resolution, grads flow)")

    # BlurSmooth test (same resolution)
    blur = BlurSmooth()
    dummy3 = torch.randn(2, 64, 128, 128)
    out3 = blur(dummy3)
    print(f"  BlurSmooth: {dummy3.shape} → {out3.shape}")
    assert out3.shape == dummy3.shape, f"BlurSmooth should preserve resolution, got {out3.shape}"
    dummy3.requires_grad_(True)
    blur(dummy3).sum().backward()
    assert dummy3.grad is not None, "BlurSmooth gradients did not flow!"
    print("  BlurSmooth OK ✓ (same resolution, grads flow)")

    # GaussianSmooth test (same resolution)
    gauss = GaussianSmooth()
    dummy4 = torch.randn(2, 64, 128, 128)
    out4 = gauss(dummy4)
    print(f"  GaussianSmooth: {dummy4.shape} → {out4.shape}")
    assert out4.shape == dummy4.shape, f"GaussianSmooth should preserve resolution, got {out4.shape}"
    dummy4.requires_grad_(True)
    gauss(dummy4).sum().backward()
    assert dummy4.grad is not None, "GaussianSmooth gradients did not flow!"
    print("  GaussianSmooth OK ✓ (same resolution, grads flow)")

    # WaveletConv test
    wconv = WaveletConv(3, 64)
    inp = torch.randn(1, 3, 640, 640)
    y = wconv(inp)
    print(f"  WaveletConv: {inp.shape} → {y.shape}")
    print("All tests passed ✓")
