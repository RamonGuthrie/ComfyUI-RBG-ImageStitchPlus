import torch
import comfy.utils
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
import numpy as np

# --- NVIDIA RTX Video Super Resolution ---
# Requires: pip install nvvfx  (RTX GPU + NVIDIA driver 570+ required)
# Always present in the upscale_methods list so saved workflows remain portable.
# Silently falls back to lanczos when nvvfx is not installed or on downscale requests.
try:
    import nvvfx as _nvvfx
    _NVVFX_AVAILABLE = True
except ImportError:
    _nvvfx = None
    _NVVFX_AVAILABLE = False

_rtx_unavail_warned = False

def _nvidia_rtx_vsr_upscale(image_cf, target_w: int, target_h: int):
    """
    Upscale a (B, C, H, W) float32 CUDA tensor via NVIDIA RTX Video Super Resolution.
    Output dimensions are set directly — no integer-scale-factor constraint.
    Returns a (B, C, H, W) float32 tensor on the original device, or None on failure.
    """
    global _rtx_unavail_warned
    src_h, src_w = image_cf.shape[2], image_cf.shape[3]
    if target_w <= src_w and target_h <= src_h:
        return None  # RTX VSR is an upscaler only

    if not _NVVFX_AVAILABLE:
        if not _rtx_unavail_warned:
            print(
                "RBG: nvidia_rtx_vsr selected but 'nvvfx' is not installed.\n"
                "  → pip install nvvfx  (RTX GPU + NVIDIA driver 570+ required)\n"
                "  Falling back to lanczos."
            )
            _rtx_unavail_warned = True
        return None

    try:
        out_w = max(8, round(target_w / 8) * 8)
        out_h = max(8, round(target_h / 8) * 8)

        with _nvvfx.VideoSuperRes(_nvvfx.effects.QualityLevel.ULTRA) as nvvfx_sr:
            nvvfx_sr.output_width = out_w
            nvvfx_sr.output_height = out_h
            nvvfx_sr.load()

            frames_chw = image_cf.cuda().contiguous()
            upscaled_frames = []
            for j in range(frames_chw.shape[0]):
                dlpack_out = nvvfx_sr.run(frames_chw[j]).image
                upscaled_frames.append(torch.from_dlpack(dlpack_out).clone())

        result = torch.stack(upscaled_frames, dim=0)  # (B, C, H, W)

        if result.shape[3] != target_w or result.shape[2] != target_h:
            result = comfy.utils.common_upscale(result, target_w, target_h, "lanczos", "disabled")

        return result.to(image_cf.device)

    except Exception as e:
        print(f"RBG: nvidia_rtx_vsr error ({e}). Falling back to lanczos.")
        return None


import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from .rbg_resampling import advanced_resample as _advanced_resample
def _legacy_advanced_resample(samples, target_width, target_height, method):
    """
    Advanced PyTorch Resampling Engine supporting Magic Kernel Sharp,
    Mitchell-Netravali, Wavelet DWT, Anti-Aliased Filters, and standard methods.
    Expects BCHW tensor in [0.0, 1.0].
    """
    _, _, h, w = samples.shape
    if h == target_height and w == target_width:
        return samples

    if method == "magic_kernel_sharp":
        down = comfy.utils.common_upscale(samples, target_width, target_height, "bicubic", "disabled")
        kernel = torch.tensor([[0, -0.125, 0], [-0.125, 1.5, -0.125], [0, -0.125, 0]], dtype=samples.dtype, device=samples.device).view(1, 1, 3, 3)
        kernel = kernel.repeat(samples.shape[1], 1, 1, 1)
        padded_down = F.pad(down, (1, 1, 1, 1), mode='replicate')
        sharpened = F.conv2d(padded_down, kernel, padding=0, groups=samples.shape[1])
        return torch.clamp(sharpened, 0.0, 1.0)

    elif method == "mitchell_netravali":
        bicubic_aa = comfy.utils.common_upscale(samples, target_width, target_height, "bicubic", "disabled")
        area_aa = comfy.utils.common_upscale(samples, target_width, target_height, "area", "disabled")
        return torch.clamp(0.66 * bicubic_aa + 0.34 * area_aa, 0.0, 1.0)

    elif method == "anti_aliased_bicubic":
        if HAS_TVF:
            out = TVF.resize(samples, [target_height, target_width], interpolation=InterpolationMode.BICUBIC, antialias=True)
        else:
            out = comfy.utils.common_upscale(samples, target_width, target_height, "bicubic", "disabled")
        return torch.clamp(out, 0.0, 1.0)

    elif method == "anti_aliased_lanczos":
        out = comfy.utils.common_upscale(samples, target_width, target_height, "lanczos", "disabled")
        return torch.clamp(out, 0.0, 1.0)

    elif method == "dwt_haar":
        scaled = samples
        while scaled.shape[2] >= target_height * 2 and scaled.shape[3] >= target_width * 2:
            scaled = F.avg_pool2d(scaled, kernel_size=2, stride=2)
        out = comfy.utils.common_upscale(scaled, target_width, target_height, "bicubic", "disabled")
        return torch.clamp(out, 0.0, 1.0)

    else:
        out = comfy.utils.common_upscale(samples, target_width, target_height, method, "disabled")
        return torch.clamp(out, 0.0, 1.0)


def _rtx_aware_upscale(image_cf, target_w: int, target_h: int, method: str):
    """
    Drop-in replacement for comfy.utils.common_upscale that adds nvidia_rtx_vsr and advanced resample methods.
    image_cf is (B, C, H, W). Returns (B, C, H, W).
    """
    if method == "nvidia_rtx_vsr":
        result = _nvidia_rtx_vsr_upscale(image_cf, target_w, target_h)
        if result is not None:
            return torch.clamp(result, 0.0, 1.0)
        res = comfy.utils.common_upscale(image_cf, target_w, target_h, "lanczos", "disabled")
        return torch.clamp(res, 0.0, 1.0)
    return _advanced_resample(image_cf, target_w, target_h, method)


class RBGImageStitchPlus:
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

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (["right", "down", "left", "up", "H_then_V_down", "H_then_V_up", "V_then_H_right", "V_then_H_left", "Grid_2x2"], {"default": "right"}),
                "keep_proportion": (["🔄 resize", "🧱 pad", "🖼️ pad_edge", "✂️ crop"], { "default": "🔄 resize" }),
                "pad_color": ("STRING", {"default": "#ffffff", "tooltip": "Color hex code (e.g., #ffffff)"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "spacing_width": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 2}),
                "spacing_color": (["white", "black", "red", "green", "blue", "custom"], {"default": "white"}),
                "custom_spacing_color": ("STRING", {"default": "#ffffff"}),
                "fill_transparent_background": ("BOOLEAN", {"default": False, "tooltip": "If true, transparent areas will be filled with the specified color."}),
                "transparent_fill_color": ("STRING", {"default": "#000000", "tooltip": "Color hex code (e.g., #000000)"}),

                # Final Resizing Options
                "final_resize_mode": (["🚫 none", "↔️ resize_longer_side", "↕️ resize_shorter_side"], { "default": "🚫 none" }),
                "final_target_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resample_filter": (s.upscale_methods, {"default": "magic_kernel_sharp", "tooltip": "magic_kernel_sharp: Magic Kernel 3 + Sharp-2013 | mitchell_netravali: B=C=1/3 spline | anti_aliased_bicubic: PyTorch AA | dwt_haar: Haar LL low-pass downscaling | nvidia_rtx_vsr: RTX AI upscaler"}),
                
                # Supersampling for anti-aliasing
                "supersample_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Upscales then downscales the final image for anti-aliasing. Factor > 1 enables it."}),
                "final_downsample_interpolation": (s.upscale_methods, {"default": "magic_kernel_sharp", "tooltip": "Interpolation for downsampling part of supersampling."}),

                # Clarity (Midtone Contrast)
                "clarity_strength": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Adjusts midtone contrast. Negative values for a dreamlike look, positive for punchy. -100=soft, +100=punchy."}),
                "sort_order": (["🎲 1-2-3", "🎲 1-3-2", "🎲 2-1-3", "🎲 2-3-1", "🎲 3-1-2", "🎲 3-2-1"], {"default": "🎲 1-2-3", "tooltip": "Order in which images will be processed and stitched together"}),
                
                # Rounded Corners - Professional Framing
                "corner_radius": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, "tooltip": "Radius for rounded corners (0-500px). Creates smooth, polished corners."}),
                "border_color": ("STRING", {"default": "#ffffff", "tooltip": "Border/frame color hex code (e.g., #ffffff for white, #000000 for black)."}),
                "outer_border_width": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Width of outer border frame around each image (0-100px). Frames each image beautifully."}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "RBG-Suite-Pack"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb + (255,)

    def _ensure_rgba(self, image_tensor):
        if image_tensor.shape[-1] == 3:
            alpha_channel = torch.ones(*image_tensor.shape[:-1], 1, device=image_tensor.device, dtype=image_tensor.dtype)
            image_tensor = torch.cat([image_tensor, alpha_channel], dim=-1)
        return image_tensor

    def get_edge_color(self, image_tensor):
        top_edge = image_tensor[:, 0, :, :]
        bottom_edge = image_tensor[:, -1, :, :]
        left_edge = image_tensor[:, :, 0, :]
        right_edge = image_tensor[:, :, -1, :]
        all_edges = torch.cat([top_edge.reshape(-1, image_tensor.shape[-1]),
                               bottom_edge.reshape(-1, image_tensor.shape[-1]),
                               left_edge.reshape(-1, image_tensor.shape[-1]),
                               right_edge.reshape(-1, image_tensor.shape[-1])], dim=0)
        mean_color = torch.mean(all_edges, dim=0) * 255.0
        color_list = mean_color.cpu().numpy().astype(int).tolist()
        if len(color_list) == 3: color_list.append(255)
        return color_list

    def pad_to_match(self, tensors, concat_dim):
        valid_tensors = [t for t in tensors if t is not None]
        if not valid_tensors: return []
        
        ref_shape = list(valid_tensors[0].shape)
        for t in valid_tensors[1:]:
            for d in range(len(ref_shape)):
                if d != concat_dim:
                    ref_shape[d] = max(ref_shape[d], t.shape[d])
        
        padded = []
        for t in valid_tensors:
            pad_spec = []
            for d in reversed(range(len(ref_shape))):
                if d == concat_dim:
                    pad_spec.extend([0, 0])
                else:
                    diff = ref_shape[d] - t.shape[d]
                    pad_spec.extend([0, diff])
            padded.append(torch.nn.functional.pad(t, pad_spec))
        return padded

    def _perform_stitch(self, img1, img2, direction, spacing_width, spacing_color, custom_spacing_color):
        if img1 is None and img2 is None: return None
        if img1 is None: return self._ensure_rgba(img2)
        if img2 is None: return self._ensure_rgba(img1)

        img1, img2 = self._ensure_rgba(img1), self._ensure_rgba(img2)

        if img1.shape[0] != img2.shape[0]:
            max_batch = max(img1.shape[0], img2.shape[0])
            if img1.shape[0] < max_batch: img1 = torch.cat([img1, img1[-1:].repeat(max_batch - img1.shape[0], 1, 1, 1)])
            if img2.shape[0] < max_batch: img2 = torch.cat([img2, img2[-1:].repeat(max_batch - img2.shape[0], 1, 1, 1)])

        spacing = None
        if spacing_width > 0:
            color_map = {"white": (255,255,255,255), "black": (0,0,0,255), "red": (255,0,0,255), "green": (0,255,0,255), "blue": (0,0,255,255)}
            color_val = self.hex_to_rgb(custom_spacing_color) if spacing_color == "custom" else color_map[spacing_color]
            
            num_channels = img1.shape[-1]
            spacing_shape = (img1.shape[0], max(img1.shape[1], img2.shape[1]), spacing_width, num_channels) if direction in ["left", "right"] else (img1.shape[0], spacing_width, max(img1.shape[2], img2.shape[2]), num_channels)
            spacing = torch.full(spacing_shape, 0.0, device=img1.device, dtype=img1.dtype)
            for j, c in enumerate(color_val):
                if j < num_channels: spacing[..., j] = c / 255.0
            if num_channels == 4: spacing[..., 3] = 1.0
        
        temp_images = [img2, img1] if direction in ["left", "up"] else [img1, img2]
        if spacing is not None: temp_images.insert(1, spacing)

        concat_dim = 2 if direction in ["left", "right"] else 1
        temp_images = self.pad_to_match(temp_images, concat_dim)
        return torch.cat(temp_images, dim=concat_dim)

    def _calculate_gaussian_kernel_size(self, sigma: float) -> int:
        """Calculates an appropriate odd kernel size for Gaussian blur."""
        radius = int(round(3 * sigma))
        kernel_s = 2 * radius + 1
        return max(1, kernel_s)

    def _apply_round_corners_and_border(self, image_tensor, radius, border_color_hex, outer_border_width=0):
        """
        Apply professional rounded corners and optional outer border to images.
        
        Args:
            image_tensor: (B, H, W, C) tensor with values 0-1
            radius: Corner radius in pixels
            border_color_hex: Hex color code for the border
            outer_border_width: Width of outer border frame (0 = no border)
        """
        B, H, W, C = image_tensor.shape
        
        # Convert tensor to numpy (0-1 range to 0-255)
        img_np = (image_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        # Parse border color
        border_rgba = self.hex_to_rgb(border_color_hex)
        
        processed_images = []
        
        for i in range(B):
            # Create RGBA image from the input
            img = Image.fromarray(img_np[i])
            if C == 3:
                img = img.convert("RGBA")
            
            # If no radius and no outer border, return as-is
            if radius == 0 and outer_border_width == 0:
                processed_img_np = np.array(img).astype(np.float32) / 255.0
                processed_images.append(torch.from_numpy(processed_img_np))
                continue
            
            # Apply rounded corners first
            if radius > 0:
                corner_mask = Image.new('L', (W, H), 0)
                corner_draw = ImageDraw.Draw(corner_mask)
                corner_draw.rounded_rectangle(
                    [(0, 0), (W - 1, H - 1)],
                    radius=radius,
                    fill=255
                )
                img.putalpha(corner_mask)
            
            # Apply outer border if specified
            if outer_border_width > 0:
                # Create new canvas with border space
                new_width = W + (outer_border_width * 2)
                new_height = H + (outer_border_width * 2)
                
                # Create border layer
                canvas = Image.new('RGBA', (new_width, new_height), border_rgba)
                
                # Create rounded corner mask for the entire bordered area
                border_mask = Image.new('L', (new_width, new_height), 0)
                border_draw = ImageDraw.Draw(border_mask)
                border_draw.rounded_rectangle(
                    [(0, 0), (new_width - 1, new_height - 1)],
                    radius=radius,  # Apply same corner radius to outer border
                    fill=255
                )
                
                # Apply rounded corner mask to border
                canvas.putalpha(border_mask)
                
                # Paste the image on top, centered
                canvas.paste(img, (outer_border_width, outer_border_width), img)
                canvas = canvas.convert('RGBA')
            else:
                canvas = img.convert('RGBA')
            
            # Convert back to tensor
            processed_img_np = np.array(canvas).astype(np.float32) / 255.0
            processed_images.append(torch.from_numpy(processed_img_np))
        
        return torch.stack(processed_images).to(image_tensor.device)

    def stitch(self, direction, sort_order, keep_proportion, pad_color, crop_position, spacing_width, spacing_color, custom_spacing_color, fill_transparent_background, transparent_fill_color, final_resize_mode, final_target_size, resample_filter, supersample_factor=1.0, final_downsample_interpolation="area", clarity_strength=0.0, corner_radius=0, border_color="#ffffffff", outer_border_width=0, image1=None, image2=None, image3=None):
        # Create ordering mapping for the 6 permutations
        order_mapping = {
            "🎲 1-2-3": [image1, image2, image3],
            "🎲 1-3-2": [image1, image3, image2],
            "🎲 2-1-3": [image2, image1, image3],
            "🎲 2-3-1": [image2, image3, image1],
            "🎲 3-1-2": [image3, image1, image2],
            "🎲 3-2-1": [image3, image2, image1]
        }
        
        # Get reordered images based on sort_order selection
        reordered_images = order_mapping.get(sort_order, [image1, image2, image3]) # Default to 1-2-3 if something goes wrong
        all_images_input = [img for img in reordered_images if img is not None and img.shape[0] > 0]
        if not all_images_input:
            raise ValueError("At least one image must be provided.")

        if len(all_images_input) == 1:
            return (self._ensure_rgba(all_images_input[0]),)

        current_batch_size = max(img.shape[0] for img in all_images_input)
        final_stitched_batch_images = []
        pbar = comfy.utils.ProgressBar(current_batch_size)

        def unify_image_for_batch_item(img_tensor, target_batch_size):
            if img_tensor is None: return None
            img_tensor = self._ensure_rgba(img_tensor)
            if img_tensor.shape[0] < target_batch_size:
                return torch.cat([img_tensor, img_tensor[-1:].repeat(target_batch_size - img_tensor.shape[0], 1, 1, 1)])
            return img_tensor

        # Unify batch sizes once for reordered images
        unified_reordered = []
        for img in reordered_images:
            unified_img = unify_image_for_batch_item(img, current_batch_size)
            unified_reordered.append(unified_img)

        for i in range(current_batch_size):
            # Extract single images for the current batch item in reorder
            batch_imgs = []
            for unified_img in unified_reordered:
                batch_img = unified_img[i:i+1] if unified_img is not None else None
                batch_imgs.append(batch_img)
            
            present_images_for_batch_item = [img for img in batch_imgs if img is not None]

            def get_processed_image(img_tensor, target_w, target_h, prop_mode):
                if img_tensor is None: return None
                resized_img = self.resize(img_tensor, target_w, target_h, prop_mode, "lanczos", 2, pad_color, crop_position)[0]
                if corner_radius > 0 or outer_border_width > 0:
                    resized_img = self._apply_round_corners_and_border(resized_img, corner_radius, border_color, outer_border_width)
                return resized_img
            stitched_image_for_this_batch_item = None
            if keep_proportion == "crop":
                if direction in ["right", "down", "left", "up"]:
                    max_dim = max(min(img.shape[1], img.shape[2]) for img in present_images_for_batch_item)
                    processed = [get_processed_image(img, max_dim, max_dim, "crop") for img in present_images_for_batch_item]
                    stitch_dir = "right" if direction in ["right", "left"] else "down"
                    if direction in ["left", "up"]: processed.reverse()
                    stitched_image_for_this_batch_item = processed[0]
                    for img in processed[1:]:
                        stitched_image_for_this_batch_item = self._perform_stitch(stitched_image_for_this_batch_item, img, stitch_dir, spacing_width, spacing_color, custom_spacing_color)
                
                elif direction == "Grid_2x2":
                    max_dim = max(min(img.shape[1], img.shape[2]) for img in present_images_for_batch_item)
                    p_img1 = get_processed_image(batch_imgs[0], max_dim, max_dim, "crop")
                    p_img2 = get_processed_image(batch_imgs[1], max_dim, max_dim, "crop")
                    p_img3 = get_processed_image(batch_imgs[2], max_dim, max_dim, "crop")
                    row1 = self._perform_stitch(p_img1, p_img2, "right", spacing_width, spacing_color, custom_spacing_color)
                    row2 = self._perform_stitch(p_img3, None, "right", spacing_width, spacing_color, custom_spacing_color)
                    stitched_image_for_this_batch_item = self._perform_stitch(row1, row2, "down", spacing_width, spacing_color, custom_spacing_color)

                else: # Compound crop
                    is_vertical_first = direction.startswith("V_then_H")
                    main_stitch_dir = "down" if is_vertical_first else "right"
                    secondary_stitch_dir = "right" if is_vertical_first else "down"
                    if direction.endswith("left"): secondary_stitch_dir = "left"
                    if direction.endswith("up"): secondary_stitch_dir = "up"
                    
                    primary_pair = [img for img in batch_imgs[:2] if img is not None]
                    if not primary_pair:
                        if batch_imgs[2] is not None:
                            stitched_image_for_this_batch_item = get_processed_image(batch_imgs[2], min(batch_imgs[2].shape[1], batch_imgs[2].shape[2]), min(batch_imgs[2].shape[1], batch_imgs[2].shape[2]), "crop")
                        else:
                            pbar.update(1)
                            continue # No images for this batch item
                    
                    elif batch_imgs[2] is None:
                        max_dim_primary = max(min(img.shape[1], img.shape[2]) for img in primary_pair)
                        p_img1 = get_processed_image(batch_imgs[0], max_dim_primary, max_dim_primary, "crop")
                        p_img2 = get_processed_image(batch_imgs[1], max_dim_primary, max_dim_primary, "crop")
                        stitched_image_for_this_batch_item = self._perform_stitch(p_img1, p_img2, main_stitch_dir, spacing_width, spacing_color, custom_spacing_color)

                    else:
                        max_dim_primary = max(min(img.shape[1], img.shape[2]) for img in primary_pair)
                        p_img1 = get_processed_image(batch_imgs[0], max_dim_primary, max_dim_primary, "crop")
                        p_img2 = get_processed_image(batch_imgs[1], max_dim_primary, max_dim_primary, "crop")
                        primary_stitch = self._perform_stitch(p_img1, p_img2, main_stitch_dir, spacing_width, spacing_color, custom_spacing_color)
                        target_dim_secondary = primary_stitch.shape[1] if is_vertical_first else primary_stitch.shape[2]
                        p_img3 = get_processed_image(batch_imgs[2], target_dim_secondary, target_dim_secondary, "crop")
                        stitched_image_for_this_batch_item = self._perform_stitch(primary_stitch, p_img3, secondary_stitch_dir, spacing_width, spacing_color, custom_spacing_color)
            else: # Logic for "stretch", "resize", "pad", "pad_edge"
                if direction in ["right", "down", "left", "up"]:
                    max_h = max(img.shape[1] for img in present_images_for_batch_item)
                    max_w = max(img.shape[2] for img in present_images_for_batch_item)
                    
                    target_h, target_w = (max_h, max_w) if keep_proportion.startswith("pad") else (max_h if direction in ["right", "left"] else 0, max_w if direction in ["down", "up"] else 0)
                    
                    processed = [get_processed_image(img, target_w, target_h, keep_proportion) for img in present_images_for_batch_item]
                    stitch_dir = "right" if direction in ["right", "left"] else "down"
                    if direction in ["left", "up"]: processed.reverse()
                    
                    stitched_image_for_this_batch_item = processed[0]
                    for img in processed[1:]:
                        stitched_image_for_this_batch_item = self._perform_stitch(stitched_image_for_this_batch_item, img, stitch_dir, spacing_width, spacing_color, custom_spacing_color)

                elif direction == "Grid_2x2":
                    max_h = max(img.shape[1] for img in present_images_for_batch_item)
                    max_w = max(img.shape[2] for img in present_images_for_batch_item)
                    p_img1 = get_processed_image(batch_imgs[0], max_w, max_h, keep_proportion)
                    p_img2 = get_processed_image(batch_imgs[1], max_w, max_h, keep_proportion)
                    p_img3 = get_processed_image(batch_imgs[2], max_w, max_h, keep_proportion)
                    row1 = self._perform_stitch(p_img1, p_img2, "right", spacing_width, spacing_color, custom_spacing_color)
                    row2 = self._perform_stitch(p_img3, None, "right", spacing_width, spacing_color, custom_spacing_color)
                    stitched_image_for_this_batch_item = self._perform_stitch(row1, row2, "down", spacing_width, spacing_color, custom_spacing_color)

                else: # Compound layouts
                    is_vertical_first = direction.startswith("V_then_H")
                    main_stitch_dir, secondary_stitch_dir = ("down", "right") if is_vertical_first else ("right", "down")
                    if direction.endswith("left"): secondary_stitch_dir = "left"
                    if direction.endswith("up"): secondary_stitch_dir = "up"

                    primary_imgs = [img for img in batch_imgs[:2] if img is not None]
                    primary_stitch = None
                    if primary_imgs:
                        max_h1 = max(img.shape[1] for img in primary_imgs)
                        max_w1 = max(img.shape[2] for img in primary_imgs)
                        
                        target_h_primary, target_w_primary = (max_h1, max_w1) if keep_proportion.startswith("pad") else ((0, max_w1) if is_vertical_first else (max_h1, 0))

                        p_img1 = get_processed_image(batch_imgs[0], target_w_primary, target_h_primary, keep_proportion)
                        p_img2 = get_processed_image(batch_imgs[1], target_w_primary, target_h_primary, keep_proportion)
                        primary_stitch = self._perform_stitch(p_img1, p_img2, main_stitch_dir, spacing_width, spacing_color, custom_spacing_color)

                    if primary_stitch is not None and batch_imgs[2] is not None:
                        target_h3, target_w3 = (primary_stitch.shape[1], 0) if is_vertical_first else (0, primary_stitch.shape[2])
                        p_img3 = get_processed_image(batch_imgs[2], target_w3, target_h3, keep_proportion)
                        stitched_image_for_this_batch_item = self._perform_stitch(primary_stitch, p_img3, secondary_stitch_dir, spacing_width, spacing_color, custom_spacing_color)
                    else:
                        stitched_image_for_this_batch_item = primary_stitch if primary_stitch is not None else batch_imgs[2]
            
            if stitched_image_for_this_batch_item is None:
                pbar.update(1)
                continue

            if fill_transparent_background and stitched_image_for_this_batch_item.shape[-1] == 4 and torch.any(stitched_image_for_this_batch_item[..., 3] < 1.0):
                fill_color_rgba = self.hex_to_rgb(transparent_fill_color)
                fill_color_tensor = torch.tensor(fill_color_rgba, device=stitched_image_for_this_batch_item.device, dtype=stitched_image_for_this_batch_item.dtype) / 255.0
                background = torch.full_like(stitched_image_for_this_batch_item, 0.0)
                background[..., :3], background[..., 3] = fill_color_tensor[:3], fill_color_tensor[3] if len(fill_color_rgba) == 4 else 1.0
                alpha, stitched_image_rgb = stitched_image_for_this_batch_item[..., 3:], stitched_image_for_this_batch_item[..., :3]
                blended_rgb = (stitched_image_rgb * alpha) + (background[..., :3] * (1 - alpha))
                stitched_image_for_this_batch_item = torch.cat([blended_rgb, torch.ones_like(alpha)], dim=-1)
            
            if final_resize_mode != "🚫 none":
                B, H, W, C = stitched_image_for_this_batch_item.shape
                if W > 0 and H > 0:
                    if final_resize_mode == "↔️ resize_longer_side":
                        ratio = final_target_size / max(W, H)
                    else:
                        ratio = final_target_size / min(W, H)
                    
                    target_width, target_height = round(W * ratio), round(H * ratio)

                    if supersample_factor > 1.0:
                        ss_width, ss_height = int(target_width * supersample_factor), int(target_height * supersample_factor)
                        temp_image = _rtx_aware_upscale(stitched_image_for_this_batch_item.movedim(-1,1), ss_width, ss_height, resample_filter).movedim(1,-1)
                        stitched_image_for_this_batch_item = _rtx_aware_upscale(temp_image.movedim(-1,1), target_width, target_height, final_downsample_interpolation).movedim(1,-1)
                    else:
                        stitched_image_for_this_batch_item = _rtx_aware_upscale(stitched_image_for_this_batch_item.movedim(-1,1), target_width, target_height, resample_filter).movedim(1,-1)
            
            elif supersample_factor > 1.0:
                ss_H, ss_W = int(stitched_image_for_this_batch_item.shape[1] * supersample_factor), int(stitched_image_for_this_batch_item.shape[2] * supersample_factor)
                stitched_image_for_this_batch_item = _rtx_aware_upscale(stitched_image_for_this_batch_item.movedim(-1,1), ss_W, ss_H, resample_filter).movedim(1,-1)

            if abs(clarity_strength) > 1e-6:
                image_bchw = stitched_image_for_this_batch_item.movedim(-1, 1)
                clarity_blur_sigma = max(1.0, min(image_bchw.shape[2], image_bchw.shape[3]) / 50.0)
                kernel_size_clarity = self._calculate_gaussian_kernel_size(clarity_blur_sigma)
                blurred_image_for_clarity = TF.gaussian_blur(image_bchw, kernel_size=(kernel_size_clarity, kernel_size_clarity), sigma=(clarity_blur_sigma, clarity_blur_sigma))
                detail_for_clarity = image_bchw - blurred_image_for_clarity
                image_bchw = image_bchw + detail_for_clarity * clarity_strength
                stitched_image_for_this_batch_item = image_bchw.clamp(0.0, 1.0).movedim(1, -1)

            final_stitched_batch_images.append(stitched_image_for_this_batch_item)
            pbar.update(1)

        if not final_stitched_batch_images:
            return (torch.empty(0),)
            
        return (torch.cat(final_stitched_batch_images, dim=0),)

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position):
        B, H, W, C = image.shape
        if width == 0 and height == 0: return (image.clone(), W, H)

        original_image = image.clone()
        target_W, target_H = width, height
        
        if keep_proportion == "crop":
            square_dim = min(W, H)
            x_crop, y_crop = {
                "center": ((W - square_dim) // 2, (H - square_dim) // 2),
                "top": ((W - square_dim) // 2, 0), "bottom": ((W - square_dim) // 2, H - square_dim),
                "left": (0, (H - square_dim) // 2), "right": (W - square_dim, (H - square_dim) // 2)
            }[crop_position]
            image = image.narrow(-3, y_crop, square_dim).narrow(-2, x_crop, square_dim)
            B, H, W, C = image.shape

        if keep_proportion == "stretch":
             new_width, new_height = target_W if target_W!=0 else W, target_H if target_H!=0 else H
        else:
            if W == 0 or H == 0: return (torch.zeros((B, target_H, target_W, C), device=image.device), target_W, target_H)
            ratio = 1.0
            if target_W == 0 and target_H != 0: ratio = target_H / H
            elif target_H == 0 and target_W != 0: ratio = target_W / W
            elif target_W != 0 and target_H != 0: ratio = min(target_W / W, target_H / H)
            new_width, new_height = round(W * ratio), round(H * ratio)

        if divisible_by > 1:
            new_width, new_height = new_width - (new_width % divisible_by), new_height - (new_height % divisible_by)

        out_image = comfy.utils.common_upscale(image.movedim(-1,1), new_width, new_height, upscale_method, crop="disabled").movedim(1,-1)
        
        if keep_proportion.startswith("pad"):
            pad_w, pad_h = (target_W if target_W!=0 else new_width), (target_H if target_H!=0 else new_height)
            if (pad_w != new_width) or (pad_h != new_height):
                pad_top, pad_left = (pad_h - new_height) // 2, (pad_w - new_width) // 2
                
                if keep_proportion == "pad":
                    color_val = self.hex_to_rgb(pad_color)
                else: # pad_edge
                    color_val = self.get_edge_color(original_image)

                color_tensor = torch.tensor(color_val, device=out_image.device, dtype=out_image.dtype).div(255.0)
                canvas = color_tensor[:C].view(1, 1, 1, C).repeat(B, pad_h, pad_w, 1)
                
                canvas[:, pad_top:pad_top+new_height, pad_left:pad_left+new_width, :] = out_image
                out_image = canvas

        return (out_image, new_width, new_height)

NODE_CLASS_MAPPINGS = { "RBGImageStitchPlus": RBGImageStitchPlus }
NODE_DISPLAY_NAME_MAPPINGS = { "RBGImageStitchPlus": "RBG Image Stitch Plus" }
