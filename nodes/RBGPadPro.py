import torch
import comfy.utils
from torchvision.transforms import functional as TF

MAX_RESOLUTION = 8192

class RBGPadPro:
    ASPECT_RATIOS = [
        "custom",
        "1:1 Square (Instagram, Facebook)",
        "2:3 Portrait (35mm Film)",
        "3:4 Portrait (Pinterest, Mobile)",
        "5:8 Portrait (Editorial/Magazine)",
        "9:16 Portrait (Instagram Stories, TikTok)",
        "9:21 Portrait (Cinematic Widescreen)",
        "4:3 Landscape (Classic TV, iPad)",
        "3:2 Landscape (35mm Film, DSLRs)",
        "8:5 Landscape (Widescreen Laptop)",
        "16:9 Landscape (HDTV, YouTube)",
        "21:9 Landscape (Cinematic Widescreen)",
    ]

    upscale_methods = ["lanczos", "bicubic", "nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_mode": (["pad", "pad_edge", "transparent_fill"], {"default": "pad"}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_feathering": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "pad_color": ("COLOR", {"default": "#FFFFFF"}),
                "image_position": (["center", "left", "right", "top", "bottom"], {"default": "center"}),
                "image_offset_x": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "image_offset_y": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "image_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fill_transparent_background": ("BOOLEAN", {"default": False}),
                "transparent_fill_color": ("COLOR", {"default": "#000000"}),
                "pad_aspect_ratio": (s.ASPECT_RATIOS, {"default": "custom"}),
                "resize_mode": (["none", "resize_longer_side", "resize_shorter_side"], { "default": "none" }),
                "target_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resample_filter": (s.upscale_methods, {"default": "bicubic"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "pad_image"
    CATEGORY = "RBG-Suite-Pack"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _ensure_rgba(self, image_tensor):
        if image_tensor.shape[-1] == 3:
            alpha_channel = torch.ones(*image_tensor.shape[:-1], 1, device=image_tensor.device, dtype=image_tensor.dtype)
            return torch.cat([image_tensor, alpha_channel], dim=-1)
        return image_tensor

    def get_edge_color(self, image_tensor):
        top_edge = image_tensor[:, 0, :, :3]
        bottom_edge = image_tensor[:, -1, :, :3]
        left_edge = image_tensor[:, :, 0, :3]
        right_edge = image_tensor[:, :, -1, :3]
        all_edges = torch.cat([
            top_edge.reshape(-1, 3), bottom_edge.reshape(-1, 3),
            left_edge.reshape(-1, 3), right_edge.reshape(-1, 3)
        ], dim=0)
        mean_color = torch.mean(all_edges, dim=0) * 255.0
        return mean_color.cpu().numpy().astype(int).tolist()

    def _calculate_gaussian_kernel_size(self, sigma: float) -> int:
        radius = int(round(3 * sigma))
        kernel_s = 2 * radius + 1
        return max(1, kernel_s)

    def pad_image(self, image, pad_mode, pad_left, pad_right, pad_top, pad_bottom, pad_feathering, pad_color, image_position, image_offset_x, image_offset_y, image_scale, fill_transparent_background, transparent_fill_color, pad_aspect_ratio, resize_mode, target_size, resample_filter, mask=None):
        image = self._ensure_rgba(image)
        B, orig_H, orig_W, C = image.shape

        aspect_pad_h = 0
        aspect_pad_w = 0
        if pad_aspect_ratio != "custom":
            ratio_str = pad_aspect_ratio.split(' ')[0]
            w_ratio, h_ratio = map(int, ratio_str.split(':'))
            target_ratio = w_ratio / h_ratio
            current_ratio = orig_W / orig_H
            
            if abs(current_ratio - target_ratio) > 1e-6:
                if current_ratio < target_ratio:
                    new_width = round(orig_H * target_ratio)
                    aspect_pad_w = new_width - orig_W
                else:
                    new_height = round(orig_W / target_ratio)
                    aspect_pad_h = new_height - orig_H

        final_width = orig_W + pad_left + pad_right + aspect_pad_w
        final_height = orig_H + pad_top + pad_bottom + aspect_pad_h

        scaled_image = image
        scaled_mask = mask
        if image_scale != 1.0:
            new_width = int(orig_W * image_scale)
            new_height = int(orig_H * image_scale)
            if new_width > 0 and new_height > 0:
                scaled_image = comfy.utils.common_upscale(image.movedim(-1,1), new_width, new_height, "lanczos", "disabled").movedim(1,-1)
                if mask is not None:
                    scaled_mask = comfy.utils.common_upscale(mask.unsqueeze(1), new_width, new_height, "bilinear", "disabled").squeeze(1)
        
        B, scaled_H, scaled_W, C = scaled_image.shape

        if scaled_mask is None:
            scaled_mask = torch.ones((B, scaled_H, scaled_W), device=image.device, dtype=torch.float32)
        else:
            scaled_mask = scaled_mask.to(image.device)
            if scaled_mask.shape[1] != scaled_H or scaled_mask.shape[2] != scaled_W:
                scaled_mask = comfy.utils.common_upscale(scaled_mask.unsqueeze(1), scaled_W, scaled_H, "bilinear", "disabled").squeeze(1)

        if pad_mode == 'transparent_fill':
            # Create a fully transparent canvas
            final_canvas = torch.zeros((B, final_height, final_width, 4), device=image.device, dtype=image.dtype)
            
            # Ensure the input image has an alpha channel
            scaled_image = self._ensure_rgba(scaled_image)
            
            # Calculate position with offset
            if image_position == 'center':
                pos_x = (final_width - scaled_W) // 2
                pos_y = (final_height - scaled_H) // 2
            elif image_position == 'left':
                pos_x = 0
                pos_y = (final_height - scaled_H) // 2
            elif image_position == 'right':
                pos_x = final_width - scaled_W
                pos_y = (final_height - scaled_H) // 2
            elif image_position == 'top':
                pos_x = (final_width - scaled_W) // 2
                pos_y = 0
            elif image_position == 'bottom':
                pos_x = (final_width - scaled_W) // 2
                pos_y = final_height - scaled_H
            
            pos_x += image_offset_x
            pos_y += image_offset_y

            # Safe copy from scaled_image to the transparent canvas
            img_start_x = max(0, -pos_x)
            img_start_y = max(0, -pos_y)
            can_start_x = max(0, pos_x)
            can_start_y = max(0, pos_y)

            copy_width = min(scaled_W - img_start_x, final_width - can_start_x)
            copy_height = min(scaled_H - img_start_y, final_height - can_start_y)

            if copy_width > 0 and copy_height > 0:
                final_canvas[:, can_start_y:can_start_y + copy_height, can_start_x:can_start_x + copy_width, :] = \
                    scaled_image[:, img_start_y:img_start_y + copy_height, img_start_x:img_start_x + copy_width, :]

            final_mask = final_canvas[..., 3].clone()

            return (final_canvas, final_mask)

        color_val = self.get_edge_color(image) if pad_mode == 'pad_edge' else self.hex_to_rgb(pad_color)
        color_tensor = torch.tensor(color_val, device=image.device, dtype=image.dtype).div(255.0)
        
        canvas = torch.zeros((B, final_height, final_width, C), device=image.device, dtype=image.dtype)
        mask_canvas = torch.zeros((B, final_height, final_width), device=image.device, dtype=torch.float32)

        if C == 4:
            canvas_rgb = color_tensor.view(1, 1, 1, 3).repeat(B, final_height, final_width, 1)
            canvas_alpha = torch.ones((B, final_height, final_width, 1), device=image.device, dtype=image.dtype)
            canvas = torch.cat([canvas_rgb, canvas_alpha], dim=-1)
        else:
            canvas = color_tensor.view(1, 1, 1, C).repeat(B, final_height, final_width, 1)

        if image_position == 'center':
            base_x = (final_width - scaled_W) // 2
            base_y = (final_height - scaled_H) // 2
        elif image_position == 'left':
            base_x = pad_left
            base_y = (final_height - scaled_H) // 2
        elif image_position == 'right':
            base_x = final_width - scaled_W - pad_right
            base_y = (final_height - scaled_H) // 2
        elif image_position == 'top':
            base_x = (final_width - scaled_W) // 2
            base_y = pad_top
        elif image_position == 'bottom':
            base_x = (final_width - scaled_W) // 2
            base_y = final_height - scaled_H - pad_bottom
        
        canvas_x = base_x + image_offset_x
        canvas_y = base_y + image_offset_y

        img_start_x = max(0, -canvas_x)
        img_start_y = max(0, -canvas_y)
        can_start_x = max(0, canvas_x)
        can_start_y = max(0, canvas_y)

        copy_width = min(scaled_W - img_start_x, final_width - can_start_x)
        copy_height = min(scaled_H - img_start_y, final_height - can_start_y)

        if copy_width > 0 and copy_height > 0:
            # Get the slices
            canvas_slice = canvas[:, can_start_y:can_start_y + copy_height, can_start_x:can_start_x + copy_width, :]
            image_slice = scaled_image[:, img_start_y:img_start_y + copy_height, img_start_x:img_start_x + copy_width, :]
            mask_slice = scaled_mask[:, img_start_y:img_start_y + copy_height, img_start_x:img_start_x + copy_width]

            # Update mask_canvas
            mask_canvas[:, can_start_y:can_start_y + copy_height, can_start_x:can_start_x + copy_width] = mask_slice

            # Blend the image onto the canvas
            blended_slice = canvas_slice * (1 - mask_slice.unsqueeze(-1)) + image_slice * mask_slice.unsqueeze(-1)

            # Put the blended slice back
            canvas[:, can_start_y:can_start_y + copy_height, can_start_x:can_start_x + copy_width, :] = blended_slice

        if pad_feathering > 0:
            sigma = pad_feathering / 10.0
            kernel_size = self._calculate_gaussian_kernel_size(sigma)
            blurred_mask = TF.gaussian_blur(mask_canvas.unsqueeze(1), kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]).squeeze(1)
            
            background_color_tensor = color_tensor
            background = background_color_tensor.view(1, 1, 1, 3).repeat(B, final_height, final_width, 1)
            
            blended_rgb = canvas[..., :3] * blurred_mask.unsqueeze(-1) + background * (1 - blurred_mask.unsqueeze(-1))
            canvas = torch.cat([blended_rgb, canvas[..., 3:]], dim=-1)
            mask_canvas = blurred_mask

        if fill_transparent_background and C == 4:
            fill_color_rgb = self.hex_to_rgb(transparent_fill_color)
            fill_color_tensor = torch.tensor(fill_color_rgb, device=image.device, dtype=image.dtype).div(255.0)
            background = fill_color_tensor.view(1, 1, 1, 3).repeat(B, final_height, final_width, 1)
            alpha = canvas[..., 3:]
            blended_rgb = canvas[..., :3] * alpha + background * (1 - alpha)
            canvas = torch.cat([blended_rgb, torch.ones_like(alpha)], dim=-1)

        # Final Resizing and Supersampling
        if resize_mode != "none":
            B, H, W, C = canvas.shape
            if W > 0 and H > 0:
                # 1. Determine final target dimensions
                if resize_mode == "resize_longer_side":
                    ratio = target_size / max(W, H)
                else: # resize_shorter_side
                    ratio = target_size / min(W, H)
                
                target_width = round(W * ratio)
                target_height = round(H * ratio)

                # Use a safe resample filter for the mask if lanczos is selected
                mask_resample_filter = resample_filter
                if mask_resample_filter == "lanczos":
                    mask_resample_filter = "bicubic"

                # No supersampling, just resize to target
                canvas = comfy.utils.common_upscale(canvas.movedim(-1,1), target_width, target_height, resample_filter, "disabled").movedim(1,-1)
                mask_canvas = comfy.utils.common_upscale(mask_canvas.unsqueeze(1), target_width, target_height, mask_resample_filter, "disabled").squeeze(1)

        return (canvas, mask_canvas)

NODE_CLASS_MAPPINGS = {
    "RBGPadPro": RBGPadPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RBGPadPro": "RBG Pad Pro"
}
