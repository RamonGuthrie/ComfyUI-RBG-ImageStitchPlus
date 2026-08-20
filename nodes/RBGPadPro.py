# License: GPL-3.0-or-later
# RBG Krea2 Smart FaceDetailer 🎨


import io
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import comfy.utils
from server import PromptServer

from .rbg_resampling import advanced_resample as _advanced_resample

MAX_RESOLUTION = 8192

# --- NVIDIA RTX Video Super Resolution ---
try:
    import nvvfx as _nvvfx
    _NVVFX_AVAILABLE = True
except ImportError:
    _nvvfx = None
    _NVVFX_AVAILABLE = False

_rtx_unavail_warned = False


def _nvidia_rtx_vsr_upscale(image_cf: torch.Tensor, target_w: int, target_h: int):
    """
    Upscale a (B, C, H, W) float32 CUDA tensor via NVIDIA RTX Video Super Resolution.
    Handles RGB (3 channels) or RGBA (4 channels) by isolating RGB and preserving Alpha.
    """
    global _rtx_unavail_warned
    src_h, src_w = image_cf.shape[2], image_cf.shape[3]
    if target_w <= src_w and target_h <= src_h:
        # RTX VSR is strictly an upscaler; cannot downscale
        return None

    if not _NVVFX_AVAILABLE:
        if not _rtx_unavail_warned:
            print(
                "RBG: nvidia_rtx_vsr selected but 'nvvfx' is not installed.\n"
                "  → pip install nvidia-vfx  (RTX GPU + NVIDIA driver 570+ required)\n"
                "  Falling back to lanczos."
            )
            _rtx_unavail_warned = True
        return None

    if not torch.cuda.is_available():
        print("RBG: CUDA not available for nvidia_rtx_vsr. Falling back to lanczos.")
        return None

    try:
        # Separate RGB (3 channels) from Alpha (4th channel) if present
        has_alpha = image_cf.shape[1] == 4
        rgb_tensor = image_cf[:, :3, :, :]
        alpha_tensor = image_cf[:, 3:4, :, :] if has_alpha else None

        target_device = image_cf.device if image_cf.device.type == "cuda" else torch.device("cuda:0")
        device_idx = target_device.index if target_device.index is not None else 0

        # NVIDIA RTX VSR output dimensions must be divisible by 2 or 8
        out_w = max(8, round(target_w / 8) * 8)
        out_h = max(8, round(target_h / 8) * 8)

        try:
            nvvfx_sr = _nvvfx.VideoSuperRes(_nvvfx.effects.QualityLevel.ULTRA, device=device_idx)
        except TypeError:
            nvvfx_sr = _nvvfx.VideoSuperRes(_nvvfx.effects.QualityLevel.ULTRA)

        with nvvfx_sr:
            nvvfx_sr.output_width = out_w
            nvvfx_sr.output_height = out_h
            nvvfx_sr.load()

            # Ensure (3, H, W) on CUDA float32
            frames_chw = rgb_tensor.to(device=target_device, dtype=torch.float32).contiguous()
            upscaled_frames = []
            for j in range(frames_chw.shape[0]):
                frame_3ch = frames_chw[j]  # Shape: (3, H, W)
                dlpack_out = nvvfx_sr.run(frame_3ch).image
                upscaled_frames.append(torch.from_dlpack(dlpack_out).clone())

        result = torch.stack(upscaled_frames, dim=0)

        # Exact pixel dimension alignment if target dimensions were not a multiple of 8
        if result.shape[3] != target_w or result.shape[2] != target_h:
            result = comfy.utils.common_upscale(result, target_w, target_h, "lanczos", "disabled")

        # Upscale and recombine Alpha channel if present
        if has_alpha and alpha_tensor is not None:
            upscaled_alpha = comfy.utils.common_upscale(alpha_tensor, target_w, target_h, "bicubic", "disabled")
            result = torch.cat([result, upscaled_alpha.to(result.device)], dim=1)

        return result.to(image_cf.device)

    except Exception as e:
        print(f"RBG: nvidia_rtx_vsr error ({e}). Falling back to lanczos.")
        return None


def _rtx_aware_upscale(image_cf: torch.Tensor, target_w: int, target_h: int, method: str):
    """
    Drop-in replacement for comfy.utils.common_upscale that integrates nvidia_rtx_vsr and advanced resamplers.
    """
    if method == "nvidia_rtx_vsr":
        result = _nvidia_rtx_vsr_upscale(image_cf, target_w, target_h)
        if result is not None:
            return torch.clamp(result, 0.0, 1.0)
        res = comfy.utils.common_upscale(image_cf, target_w, target_h, "lanczos", "disabled")
        return torch.clamp(res, 0.0, 1.0)
    return _advanced_resample(image_cf, target_w, target_h, method)


class RBGPadPro:
    ASPECT_RATIOS = [
        "custom",
        "⏹️ 1:1 Square (Instagram, Facebook)",
        "📸 2:3 Portrait (35mm Film)",
        "📱 3:4 Portrait (Pinterest, Mobile)",
        "📰 5:8 Portrait (Editorial/Magazine)",
        "📲 9:16 Portrait (Instagram Stories, TikTok)",
        "🎥 9:21 Portrait (Cinematic Widescreen)",
        "🖥️ 4:3 Landscape (Classic TV, iPad)",
        "📷 3:2 Landscape (35mm Film, DSLRs)",
        "💻 8:5 Landscape (Widescreen Laptop)",
        "📺 16:9 Landscape (HDTV, YouTube)",
        "🎞️ 21:9 Landscape (Cinematic Widescreen)",
    ]

    upscale_methods = [
        "nvidia_rtx_vsr",
        "lanczos",
        "bicubic",
        "nearest-exact",
        "bilinear",
        "area",
        "magic_kernel_sharp",
        "mitchell_netravali",
        "anti_aliased_bicubic",
        "anti_aliased_lanczos",
        "dwt_haar",
    ]

    DIVISIBILITY_OPTIONS = ["None", "2", "4", "8", "16", "32", "64", "128", "256"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_mode": (["🧱 pad", "🖼️ pad_edge", "🧬 pad_edge_pixel", "🪟 transparent_fill"], {"default": "🧱 pad"}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "mask_blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "pad_color": ("STRING", {"default": "#00FF00"}),
                "image_position": (["center", "left", "right", "top", "bottom"], {"default": "center"}),
                "image_offset_x": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "image_offset_y": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "image_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fill_transparent_background": ("BOOLEAN", {"default": False}),
                "transparent_fill_color": ("STRING", {"default": "#7F7F7F"}),
                "pad_aspect_ratio": (s.ASPECT_RATIOS, {"default": "custom"}),
                "resize_mode": (["🚫 none", "↔️ resize_longer_side", "↕️ resize_shorter_side"], {"default": "🚫 none"}),
                "target_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resample_filter": (s.upscale_methods, {"default": "magic_kernel_sharp", "tooltip": "magic_kernel_sharp: Costella MK3 + Sharp-2013 | mitchell_netravali: B=C=1/3 Spline | anti_aliased_bicubic: PyTorch AA Bicubic | dwt_haar: Haar LL DWT Downscaling | nvidia_rtx_vsr: AI TensorRT Upscaling"}),
                "auto_crop_aspect_ratio": ("BOOLEAN", {"default": False, "label_on": "✂️ Auto-Crop to Ratio", "label_off": "🚫 Auto-Crop to Ratio", "tooltip": "Crops the image to match the selected pad_aspect_ratio instead of expanding the canvas with padding."}),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "flip_horizontal": ("BOOLEAN", {"default": False, "label_on": "↔️ Flip Horizontal", "label_off": "🚫 Flip Horizontal"}),
                "image_rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.01}),
                "BorderCrop_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider"}),
                "BorderCrop_color": ("STRING", {"default": "#000000"}),
                "pad_noise_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider", "tooltip": "Strength of noise added to the padded area. Useful for outpainting."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the random noise generation."}),
                "noise_mode": (["Color (Default)", "Greyscale", "Greyscale + Mask"], {"default": "Color (Default)"}),
                "divisibility": (s.DIVISIBILITY_OPTIONS, {"default": "None", "tooltip": "Enforce output dimensions to be divisible by this amount (e.g., 8, 16, 32 for model compatibility)"}),
                "show_preview": ("BOOLEAN", {"default": False, "label_on": "Preview Enabled", "label_off": "Preview Disabled"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "pad_image"
    CATEGORY = "RBG-Suite-Pack"

    def hex_to_rgb(self, hex_color: str):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def get_image_info(self, image_tensor: torch.Tensor):
        if image_tensor is None:
            return "Output: No image"
        batch_size, height, width, _ = image_tensor.shape
        num_elements = image_tensor.numel()
        element_size = image_tensor.element_size()
        total_size_mb = (num_elements * element_size) / (1024 * 1024)
        return f"Output: {batch_size} x {width} x {height} | {total_size_mb:.2f}MB"

    def prepare_image_for_display(self, image_tensor: torch.Tensor):
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            img_np = image_tensor.squeeze(0).detach().cpu().numpy()
        else:
            img_np = image_tensor.detach().cpu().numpy()

        img_pil = Image.fromarray((img_np.clip(0, 1) * 255).astype(np.uint8))
        img_pil.thumbnail((512, 512), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    def _ensure_rgba(self, image_tensor: torch.Tensor):
        if image_tensor.shape[-1] == 3:
            alpha_channel = torch.ones(*image_tensor.shape[:-1], 1, device=image_tensor.device, dtype=image_tensor.dtype)
            return torch.cat([image_tensor, alpha_channel], dim=-1)
        return image_tensor

    def get_edge_color(self, image_tensor: torch.Tensor):
        edges = torch.cat([
            image_tensor[:, 0, :, :3].reshape(-1, 3), image_tensor[:, -1, :, :3].reshape(-1, 3),
            image_tensor[:, :, 0, :3].reshape(-1, 3), image_tensor[:, :, -1, :3].reshape(-1, 3)
        ], dim=0)
        return torch.mean(edges, dim=0).mul(255.0).cpu().numpy().astype(int).tolist()

    def _calculate_gaussian_kernel_size(self, sigma: float) -> int:
        return max(1, 2 * int(round(3 * sigma)) + 1)

    def pad_image(self, image, pad_mode, pad_left, pad_right, pad_top, pad_bottom, mask_blur_sigma, pad_color, pad_noise_strength, noise_seed, noise_mode, image_position, image_offset_x, image_offset_y, image_scale, fill_transparent_background, transparent_fill_color, pad_aspect_ratio, resize_mode, target_size, resample_filter, auto_crop_aspect_ratio, invert_mask, flip_horizontal, image_rotation, BorderCrop_threshold=0.0, BorderCrop_color="#000000", divisibility="None", show_preview=False, mask=None):
        pad_mode = pad_mode.split(" ", 1)[-1] if " " in pad_mode else pad_mode
        resize_mode = resize_mode.split(" ", 1)[-1] if " " in resize_mode else resize_mode
        image = self._ensure_rgba(image)

        div_factor = int(divisibility) if divisibility != "None" else 1

        # Border cropping
        if BorderCrop_threshold > 0:
            check_channels = 3 if image.shape[-1] > 3 else image.shape[-1]
            try:
                border_rgb = self.hex_to_rgb(BorderCrop_color)
            except Exception:
                border_rgb = (0, 0, 0)

            border_tensor = torch.tensor(border_rgb, device=image.device, dtype=image.dtype).div(255.0).view(1, 1, 1, 3)
            img_check = image[..., :check_channels]
            if img_check.shape[-1] == 1:
                border_tensor = (border_tensor[..., 0] * 0.299 + border_tensor[..., 1] * 0.587 + border_tensor[..., 2] * 0.114).unsqueeze(-1)

            mask_threshold = (torch.abs(img_check - border_tensor) > BorderCrop_threshold).any(dim=-1).any(dim=0)
            coords = torch.nonzero(mask_threshold)
            if len(coords) > 0:
                y_min, x_min = coords.min(dim=0).values.tolist()
                y_max, x_max = coords.max(dim=0).values.tolist()
                image = image[:, y_min:y_max+1, x_min:x_max+1, :]
                if mask is not None:
                    mask = mask[:, y_min:y_max+1, x_min:x_max+1]

        if flip_horizontal:
            image = torch.flip(image, [2])
            if mask is not None:
                mask = torch.flip(mask, [2])

        if image_rotation != 0.0:
            image = image.permute(0, 3, 1, 2)
            image = TF.rotate(image, image_rotation, interpolation=InterpolationMode.BILINEAR, expand=True, fill=0)
            image = image.permute(0, 2, 3, 1)
            if mask is not None:
                mask = mask.unsqueeze(1)
                mask = TF.rotate(mask, image_rotation, interpolation=InterpolationMode.BILINEAR, expand=True, fill=0)
                mask = mask.squeeze(1)

        B, orig_H, orig_W, C = image.shape

        # Parse Aspect Ratio
        target_ratio = None
        if pad_aspect_ratio != "custom":
            ratio_part = next((part for part in pad_aspect_ratio.split(' ') if ':' in part), None)
            if ratio_part is None:
                raise ValueError(f"Invalid ratio: '{pad_aspect_ratio}'")
            w_ratio, h_ratio = map(int, ratio_part.split(':'))
            target_ratio = w_ratio / h_ratio

        # Auto-Crop to Aspect Ratio Logic
        crop_rect = (0, 0, orig_W, orig_H)
        if auto_crop_aspect_ratio and target_ratio is not None:
            current_ratio = orig_W / orig_H
            if abs(current_ratio - target_ratio) > 1e-6:
                if current_ratio > target_ratio:
                    crop_H = orig_H
                    crop_W = int(round(orig_H * target_ratio))
                    if image_position == 'left':
                        crop_x = 0
                    elif image_position == 'right':
                        crop_x = orig_W - crop_W
                    else:
                        crop_x = (orig_W - crop_W) // 2
                    crop_y = 0
                else:
                    crop_W = orig_W
                    crop_H = int(round(orig_W / target_ratio))
                    if image_position == 'top':
                        crop_y = 0
                    elif image_position == 'bottom':
                        crop_y = orig_H - crop_H
                    else:
                        crop_y = (orig_H - crop_H) // 2
                    crop_x = 0
                crop_rect = (crop_x, crop_y, crop_W, crop_H)

        crop_W, crop_H = crop_rect[2], crop_rect[3]

        # Pad to Aspect Ratio Logic (Only when auto-crop is OFF)
        aspect_pad_h, aspect_pad_w = 0, 0
        if pad_aspect_ratio != "custom" and not auto_crop_aspect_ratio and target_ratio is not None:
            current_ratio = crop_W / crop_H
            if abs(current_ratio - target_ratio) > 1e-6:
                if current_ratio < target_ratio:
                    aspect_pad_w = round(crop_H * target_ratio) - crop_W
                else:
                    aspect_pad_h = round(crop_W / target_ratio) - crop_H

        final_width = crop_W + pad_left + pad_right + aspect_pad_w
        final_height = crop_H + pad_top + pad_bottom + aspect_pad_h

        # Enforce divisibility upfront if not resizing afterwards
        if resize_mode == "none" and div_factor > 1:
            final_width = ((final_width + div_factor - 1) // div_factor) * div_factor
            final_height = ((final_height + div_factor - 1) // div_factor) * div_factor

        scaled_image = image
        scaled_crop_rect = crop_rect
        if image_scale != 1.0:
            target_scale_w = int(round(orig_W * image_scale))
            target_scale_h = int(round(orig_H * image_scale))
            scaled_image = _rtx_aware_upscale(image.movedim(-1, 1), target_scale_w, target_scale_h, resample_filter).movedim(1, -1)
            scaled_crop_rect = tuple(int(round(x * image_scale)) for x in crop_rect)
            if mask is not None:
                scaled_mask = comfy.utils.common_upscale(mask.unsqueeze(1), target_scale_w, target_scale_h, "bilinear", "disabled").squeeze(1)

        B, scaled_H_full, scaled_W_full, C = scaled_image.shape
        scaled_crop_W, scaled_crop_H = scaled_crop_rect[2], scaled_crop_rect[3]

        scaled_mask = mask if mask is not None else torch.ones((B, scaled_H_full, scaled_W_full), device=image.device)
        if mask is not None and (scaled_mask.shape[1] != scaled_H_full or scaled_mask.shape[2] != scaled_W_full):
            scaled_mask = comfy.utils.common_upscale(scaled_mask.unsqueeze(1), scaled_W_full, scaled_H_full, "bilinear", "disabled").squeeze(1)

        canvas = torch.zeros((B, final_height, final_width, C), device=image.device, dtype=image.dtype)
        mask_canvas = torch.zeros((B, final_height, final_width), device=image.device, dtype=torch.float32)
        region_mask = torch.zeros((B, final_height, final_width), device=image.device, dtype=torch.float32)

        position_map = {
            'center': ((final_width - scaled_crop_W) // 2, (final_height - scaled_crop_H) // 2),
            'left': (0, (final_height - scaled_crop_H) // 2),
            'right': (final_width - scaled_crop_W, (final_height - scaled_crop_H) // 2),
            'top': ((final_width - scaled_crop_W) // 2, 0),
            'bottom': ((final_width - scaled_crop_W) // 2, final_height - scaled_crop_H)
        }
        viewport_pos_x, viewport_pos_y = position_map.get(image_position, position_map['center'])

        src_start_x = scaled_crop_rect[0] - image_offset_x
        src_start_y = scaled_crop_rect[1] - image_offset_y

        dst_start_x = viewport_pos_x
        dst_start_y = viewport_pos_y

        copy_width = scaled_crop_W
        copy_height = scaled_crop_H

        if src_start_x < 0:
            dst_start_x += -src_start_x
            copy_width += src_start_x
            src_start_x = 0

        if src_start_y < 0:
            dst_start_y += -src_start_y
            copy_height += src_start_y
            src_start_y = 0

        if dst_start_x < 0:
            src_start_x += -dst_start_x
            copy_width += dst_start_x
            dst_start_x = 0

        if dst_start_y < 0:
            src_start_y += -dst_start_y
            copy_height += dst_start_y
            dst_start_y = 0

        copy_width = int(min(copy_width, scaled_W_full - src_start_x, final_width - dst_start_x))
        copy_height = int(min(copy_height, scaled_H_full - src_start_y, final_height - dst_start_y))
        copy_width = max(0, copy_width)
        copy_height = max(0, copy_height)

        if pad_mode not in ['pad_edge_pixel', 'transparent_fill']:
            color_val = self.get_edge_color(image) if pad_mode == 'pad_edge' else self.hex_to_rgb(pad_color)
            color_tensor = torch.tensor(color_val, device=image.device, dtype=image.dtype).div(255.0)
            canvas[..., :3] = color_tensor.view(1, 1, 1, 3)
            canvas[..., 3] = 1.0

        if copy_width > 0 and copy_height > 0:
            img_slice = (slice(src_start_y, src_start_y + copy_height), slice(src_start_x, src_start_x + copy_width))
            can_slice = (slice(dst_start_y, dst_start_y + copy_height), slice(dst_start_x, dst_start_x + copy_width))
            region_mask[:, can_slice[0], can_slice[1]] = 1.0

            if pad_mode == 'pad_edge_pixel':
                canvas[:, can_slice[0], can_slice[1], :] = scaled_image[:, img_slice[0], img_slice[1], :]
                mask_canvas[:, can_slice[0], can_slice[1]] = scaled_mask[:, img_slice[0], img_slice[1]]
                img_end_y, img_end_x = dst_start_y + copy_height, dst_start_x + copy_width
                if dst_start_y > 0:
                    canvas[:, 0:dst_start_y, dst_start_x:img_end_x, :] = canvas[:, dst_start_y:dst_start_y+1, dst_start_x:img_end_x, :].repeat(1, dst_start_y, 1, 1)
                if img_end_y < final_height:
                    canvas[:, img_end_y:final_height, dst_start_x:img_end_x, :] = canvas[:, img_end_y-1:img_end_y, dst_start_x:img_end_x, :].repeat(1, final_height-img_end_y, 1, 1)
                if dst_start_x > 0:
                    canvas[:, :, 0:dst_start_x, :] = canvas[:, :, dst_start_x:dst_start_x+1, :].repeat(1, 1, dst_start_x, 1)
                if img_end_x < final_width:
                    canvas[:, :, img_end_x:final_width, :] = canvas[:, :, img_end_x-1:img_end_x, :].repeat(1, 1, final_width-img_end_x, 1)
            elif pad_mode == 'transparent_fill':
                canvas[:, can_slice[0], can_slice[1], :] = scaled_image[:, img_slice[0], img_slice[1], :]
                mask_canvas[:, can_slice[0], can_slice[1]] = scaled_image[:, img_slice[0], img_slice[1], 3]
            else:
                img_part = scaled_image[:, img_slice[0], img_slice[1], :]
                mask_part = scaled_mask[:, img_slice[0], img_slice[1]].unsqueeze(-1)
                canvas[:, can_slice[0], can_slice[1], :] = img_part * mask_part + canvas[:, can_slice[0], can_slice[1], :] * (1 - mask_part)
                mask_canvas[:, can_slice[0], can_slice[1]] = scaled_mask[:, img_slice[0], img_slice[1]]

        if mask_blur_sigma > 0:
            kernel_size = self._calculate_gaussian_kernel_size(mask_blur_sigma)
            if kernel_size > 1:
                mask_canvas = TF.gaussian_blur(mask_canvas.unsqueeze(1), kernel_size=[kernel_size, kernel_size], sigma=[mask_blur_sigma, mask_blur_sigma]).squeeze(1)

        if fill_transparent_background and C == 4:
            fill_color_tensor = torch.tensor(self.hex_to_rgb(transparent_fill_color), device=image.device, dtype=image.dtype).div(255.0)
            alpha = canvas[..., 3:]
            canvas[..., :3] = canvas[..., :3] * alpha + fill_color_tensor.view(1, 1, 1, 3) * (1 - alpha)
            canvas[..., 3] = 1.0

        # Resizing mode with divisibility alignment (defaulting to 8 if None)
        if resize_mode != "none":
            B, H, W, C = canvas.shape
            if W > 0 and H > 0:
                align_factor = max(8, div_factor) if div_factor > 1 else 8
                ratio = target_size / max(W, H) if resize_mode == "resize_longer_side" else target_size / min(W, H)
                target_width = max(align_factor, (round(W * ratio) // align_factor) * align_factor)
                target_height = max(align_factor, (round(H * ratio) // align_factor) * align_factor)

                if target_width > 0 and target_height > 0:
                    COMFY_NATIVE_MODES = {"nearest-exact", "bilinear", "bicubic", "area", "lanczos"}
                    mask_resample_filter = resample_filter if resample_filter in COMFY_NATIVE_MODES else "bicubic"

                    canvas = _rtx_aware_upscale(canvas.movedim(-1, 1), target_width, target_height, resample_filter).movedim(1, -1)
                    mask_canvas = comfy.utils.common_upscale(mask_canvas.unsqueeze(1), target_width, target_height, mask_resample_filter, "disabled").squeeze(1)
                    region_mask = comfy.utils.common_upscale(region_mask.unsqueeze(1), target_width, target_height, "nearest-exact", "disabled").squeeze(1)

        # Pad Noise with local Generator to preserve global random states
        if pad_noise_strength > 0:
            B, H, W, C = canvas.shape
            gen_device = canvas.device if canvas.device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=gen_device)
            generator.manual_seed(noise_seed)

            if "Greyscale" in noise_mode:
                greyscale_noise = torch.randn((B, H, W, 1), generator=generator, device=gen_device, dtype=canvas.dtype).to(canvas.device) * pad_noise_strength
                noise_for_image = greyscale_noise.repeat(1, 1, 1, 3)
            else:
                noise_for_image = torch.randn((B, H, W, 3), generator=generator, device=gen_device, dtype=canvas.dtype).to(canvas.device) * pad_noise_strength
                greyscale_noise = None

            if noise_mode == "Greyscale + Mask":
                padding_mask_for_image = (1.0 - (region_mask * mask_canvas)).clamp(0.0, 1.0).unsqueeze(-1)
            else:
                padding_mask_for_image = (1.0 - mask_canvas).unsqueeze(-1)

            canvas[..., :3] = torch.clamp(canvas[..., :3] + noise_for_image * padding_mask_for_image, 0.0, 1.0)

            if noise_mode == "Greyscale + Mask" and greyscale_noise is not None:
                noise_application_mask_for_mask = padding_mask_for_image.squeeze(-1)
                mask_canvas = torch.clamp(mask_canvas + greyscale_noise.squeeze(-1) * noise_application_mask_for_mask, 0.0, 1.0)

        if invert_mask:
            mask_canvas = 1.0 - mask_canvas

        canvas = canvas[..., :3]

        if show_preview:
            try:
                preview_image_tensor = canvas[0:1]
                preview_image = self.prepare_image_for_display(preview_image_tensor)
                PromptServer.instance.send_sync("rbg_pad_pro_preview", {"image": preview_image})
            except Exception as e:
                print(f"RBGPadPro Warning: Could not send preview image. {e}")

        image_info = self.get_image_info(canvas)
        return {"ui": {"text": [image_info]}, "result": (canvas, mask_canvas)}


NODE_CLASS_MAPPINGS = {"RBGPadPro": RBGPadPro}
NODE_DISPLAY_NAME_MAPPINGS = {"RBGPadPro": "RBG Pad Pro"}