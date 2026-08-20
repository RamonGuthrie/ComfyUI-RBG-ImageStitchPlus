"""High-quality resamplers shared by the RBG image layout nodes."""

import torch
import torch.nn.functional as F
import comfy.utils


def _magic_kernel_3(x: torch.Tensor):
    """John Costella's Magic Kernel 3 (quadratic cardinal B-spline)."""
    ax = x.abs()
    return torch.where(ax < 0.5, 0.75 - ax.square(), torch.where(ax < 1.5, 0.5 * (1.5 - ax).square(), torch.zeros_like(ax)))


def _mitchell_netravali(x: torch.Tensor):
    """Mitchell-Netravali BC spline with B=C=1/3."""
    ax = x.abs()
    b = c = 1.0 / 3.0
    inner = ((12 - 9 * b - 6 * c) * ax**3 + (-18 + 12 * b + 6 * c) * ax**2 + (6 - 2 * b)) / 6
    outer = ((-b - 6 * c) * ax**3 + (6 * b + 30 * c) * ax**2 + (-12 * b - 48 * c) * ax + (8 * b + 24 * c)) / 6
    return torch.where(ax < 1, inner, torch.where(ax < 2, outer, torch.zeros_like(ax)))


def _resample_axis(samples: torch.Tensor, target_size: int, axis: int, kernel, support: float):
    """Resample one spatial axis with a normalized, scale-aware continuous kernel."""
    source_size = samples.shape[axis]
    if source_size == target_size:
        return samples
    scale = target_size / source_size
    kernel_scale = min(scale, 1.0)
    radius = support / kernel_scale
    taps = int(torch.ceil(torch.tensor(radius * 2)).item())
    destination = torch.arange(target_size, dtype=samples.dtype, device=samples.device)
    center = (destination + 0.5) / scale - 0.5
    first = torch.floor(center - radius + 1).to(torch.long)
    indices = first[:, None] + torch.arange(taps, device=samples.device)[None, :]
    weights = kernel((center[:, None] - indices) * kernel_scale) * kernel_scale
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(samples.dtype).eps)
    indices = indices.clamp(0, source_size - 1)
    if axis == -1 or axis == 3:
        return (samples[..., indices] * weights[None, None, None, :, :]).sum(dim=-1)
    return _resample_axis(samples.transpose(-2, -1), target_size, -1, kernel, support).transpose(-2, -1)


def _separable_resample(samples: torch.Tensor, target_width: int, target_height: int, kernel, support: float):
    return _resample_axis(_resample_axis(samples, target_width, -1, kernel, support), target_height, -2, kernel, support)


def _magic_kernel_sharp(samples: torch.Tensor, target_width: int, target_height: int):
    """Magic Kernel 3 followed by Costella's Sharp-2013 operator {-1/4, 3/2, -1/4}."""
    out = _separable_resample(samples, target_width, target_height, _magic_kernel_3, 1.5)
    sharp = torch.tensor((-0.25, 1.5, -0.25), dtype=samples.dtype, device=samples.device)
    channels = samples.shape[1]
    horizontal = sharp.view(1, 1, 1, 3).repeat(channels, 1, 1, 1)
    vertical = sharp.view(1, 1, 3, 1).repeat(channels, 1, 1, 1)
    out = F.conv2d(F.pad(out, (1, 1, 0, 0), mode="replicate"), horizontal, groups=channels)
    return F.conv2d(F.pad(out, (0, 0, 1, 1), mode="replicate"), vertical, groups=channels)


def _haar_ll_downsample(samples: torch.Tensor, target_width: int, target_height: int):
    """Repeated Haar analysis retaining LL band, followed by antialiased bicubic."""
    out = samples
    while out.shape[-2] >= target_height * 2 and out.shape[-1] >= target_width * 2:
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
    return F.interpolate(out, size=(target_height, target_width), mode="bicubic", align_corners=False, antialias=True)


def advanced_resample(samples: torch.Tensor, target_width: int, target_height: int, method: str):
    """Dispatch standard and custom filters without silent substitution."""
    if samples.shape[-2:] == (target_height, target_width):
        return samples
    if method == "magic_kernel_sharp":
        out = _magic_kernel_sharp(samples, target_width, target_height)
    elif method == "mitchell_netravali":
        out = _separable_resample(samples, target_width, target_height, _mitchell_netravali, 2.0)
    elif method == "anti_aliased_bicubic":
        out = F.interpolate(samples, size=(target_height, target_width), mode="bicubic", align_corners=False, antialias=True)
    elif method == "anti_aliased_lanczos":
        out = comfy.utils.common_upscale(samples, target_width, target_height, "lanczos", "disabled")
    elif method == "dwt_haar":
        out = _haar_ll_downsample(samples, target_width, target_height) if target_width < samples.shape[-1] or target_height < samples.shape[-2] else F.interpolate(samples, size=(target_height, target_width), mode="bicubic", align_corners=False)
    else:
        out = comfy.utils.common_upscale(samples, target_width, target_height, method, "disabled")
    return out.clamp(0.0, 1.0)