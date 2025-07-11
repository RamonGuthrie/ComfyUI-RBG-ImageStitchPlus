import torch
import comfy.utils
from torchvision.transforms import functional as TF

MAX_RESOLUTION = 8192

class RBGPadPro:
    ASPECT_RATIOS = [
        "custom", "1:1 square", "2:3 portrait", "3:4 portrait", "5:8 portrait", 
        "9:16 portrait", "9:21 portrait", "4:3 landscape", "3:2 landscape", 
        "8:5 landscape", "16:9 landscape", "21:9 landscape"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_mode": (["pad", "pad_edge"], {"default": "pad"}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "pad_feathering": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "pad_color": ("COLOR", {"default": "#FFFFFF"}),
                "image_position": (["center", "left", "right", "top", "bottom"], {"default": "center"}),
                "image_offset_x": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "image_offset_y": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "fill_transparent_background": ("BOOLEAN", {"default": False}),
                "transparent_fill_color": ("COLOR", {"default": "#000000"}),
                "pad_aspect_ratio": (s.ASPECT_RATIOS, {"default": "custom"}),
            },
            "optional": {
                "mask": ("MASK",),
                "target_width": ("INT", {"default": None, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "target_height": ("INT", {"default": None, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
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

    def pad_image(self, image, pad_mode, pad_left, pad_right, pad_top, pad_bottom, pad_feathering, pad_color, image_position, image_offset_x, image_offset_y, fill_transparent_background, transparent_fill_color, pad_aspect_ratio, mask=None, target_width=None, target_height=None):
        image = self._ensure_rgba(image)
        B, H, W, C = image.shape

        if mask is None:
            mask = torch.ones((B, H, W), device=image.device, dtype=torch.float32)
        else:
            mask = mask.to(image.device)

        aspect_pad_h = 0
        aspect_pad_w = 0

        if pad_aspect_ratio != "custom":
            ratio_str = pad_aspect_ratio.split(' ')[0]
            w_ratio, h_ratio = map(int, ratio_str.split(':'))
            target_ratio = w_ratio / h_ratio
            current_ratio = W / H
            
            if abs(current_ratio - target_ratio) > 1e-6:
                if current_ratio < target_ratio:
                    new_width = round(H * target_ratio)
                    aspect_pad_w = new_width - W
                else:
                    new_height = round(W / target_ratio)
                    aspect_pad_h = new_height - H

        final_width = W + pad_left + pad_right + aspect_pad_w
        final_height = H + pad_top + pad_bottom + aspect_pad_h

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
            base_x = pad_left + aspect_pad_w // 2
            base_y = pad_top + aspect_pad_h // 2
        elif image_position == 'left':
            base_x = pad_left
            base_y = pad_top + aspect_pad_h // 2
        elif image_position == 'right':
            base_x = final_width - W - pad_right
            base_y = pad_top + aspect_pad_h // 2
        elif image_position == 'top':
            base_x = pad_left + aspect_pad_w // 2
            base_y = pad_top
        elif image_position == 'bottom':
            base_x = pad_left + aspect_pad_w // 2
            base_y = final_height - H - pad_bottom
        
        canvas_x = base_x + image_offset_x
        canvas_y = base_y + image_offset_y

        canvas[:, canvas_y:canvas_y + H, canvas_x:canvas_x + W, :] = image
        mask_canvas[:, canvas_y:canvas_y + H, canvas_x:canvas_x + W] = mask

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

        if target_width and target_height and target_width > 0 and target_height > 0:
            canvas = comfy.utils.common_upscale(canvas.movedim(-1,1), target_width, target_height, "lanczos", "disabled").movedim(1,-1)
            mask_canvas = comfy.utils.common_upscale(mask_canvas.unsqueeze(1), target_width, target_height, "bilinear", "disabled").squeeze(1)

        return (canvas, mask_canvas)

NODE_CLASS_MAPPINGS = {
    "RBGPadPro": RBGPadPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RBGPadPro": "RBG Pad Pro"
}
