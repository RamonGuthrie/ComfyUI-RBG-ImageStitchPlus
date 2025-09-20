import torch
import comfy.utils
from torchvision.transforms import functional as TF

class RBGImageStitchPlus:
    # Added "mitchell" to the list of upscale methods as requested.
    upscale_methods = ["lanczos", "bicubic", "nearest-exact", "bilinear", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (["right", "down", "left", "up", "H_then_V_down", "H_then_V_up", "V_then_H_right", "V_then_H_left", "Grid_2x2"], {"default": "right"}),
                "keep_proportion": (["resize", "pad", "pad_edge", "crop"], { "default": "resize" }),
                "pad_color": ("STRING", {"default": "#FFFFFF", "tooltip": "Color to use for padding (R,G,B)."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "spacing_width": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 2}),
                "spacing_color": (["white", "black", "red", "green", "blue", "custom"], {"default": "white"}),
                "custom_spacing_color": ("STRING", {"default": "#FFFFFF"}),
                "fill_transparent_background": ("BOOLEAN", {"default": False, "tooltip": "If true, transparent areas will be filled with the specified color."}),
                "transparent_fill_color": ("STRING", {"default": "#000000", "tooltip": "Color to fill transparent areas (R,G,B). Only used if 'fill_transparent_background' is true."}),
                
                # Final Resizing Options
                "final_resize_mode": (["none", "resize_longer_side", "resize_shorter_side"], { "default": "none" }),
                "final_target_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resample_filter": (s.upscale_methods, {"default": "bicubic", "tooltip": "Interpolation for general resizing and the upscaling part of supersampling."}),
                
                # Supersampling for anti-aliasing
                "supersample_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Upscales then downscales the final image for anti-aliasing. Factor > 1 enables it."}),
                "final_downsample_interpolation": (s.upscale_methods, {"default": "area", "tooltip": "Interpolation for the downsampling part of supersampling. 'area' is often best for this."}),

                # Clarity (Midtone Contrast)
                "clarity_strength": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Adjusts midtone contrast. Negative values for a dreamlike look, positive for punchy. -100=soft, +100=punchy."}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "RBG/ImageStitchPlus"

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
    def stitch(self, direction, keep_proportion, pad_color, crop_position, spacing_width, spacing_color, custom_spacing_color, fill_transparent_background, transparent_fill_color, final_resize_mode, final_target_size, resample_filter, supersample_factor=1.0, final_downsample_interpolation="area", clarity_strength=0.0, image1=None, image2=None, image3=None):
        all_images_input = [img for img in [image1, image2, image3] if img is not None and img.shape[0] > 0]
        if not all_images_input: raise ValueError("At least one image must be provided.")
        if len(all_images_input) == 1: return (self._ensure_rgba(all_images_input[0]),)

        # Initialize progress bar with 5 potential major steps
        pbar = comfy.utils.ProgressBar(5)

        current_batch_size = max(img.shape[0] for img in all_images_input)
        def unify_image(img_tensor):
            if img_tensor is None: return None
            img_tensor = self._ensure_rgba(img_tensor)
            if img_tensor.shape[0] < current_batch_size:
                return torch.cat([img_tensor, img_tensor[-1:].repeat(current_batch_size - img_tensor.shape[0], 1, 1, 1)])
            return img_tensor

        image1, image2, image3 = unify_image(image1), unify_image(image2), unify_image(image3)
        present_images = [img for img in [image1, image2, image3] if img is not None]
        pbar.update(1) # Step 1: Image unification and prep complete

        def get_processed_image(img_tensor, target_w, target_h, prop_mode):
            if img_tensor is None: return None
            return self.resize(img_tensor, target_w, target_h, prop_mode, "lanczos", 2, pad_color, crop_position)[0]

        stitched_image = None
        if keep_proportion == "crop":
            if direction in ["right", "down", "left", "up"]:
                max_dim = max(min(img.shape[1], img.shape[2]) for img in present_images)
                processed = [get_processed_image(img, max_dim, max_dim, "crop") for img in present_images]
                stitch_dir = "right" if direction in ["right", "left"] else "down"
                if direction in ["left", "up"]: processed.reverse()
                stitched_image = processed[0]
                for i in range(1, len(processed)):
                    stitched_image = self._perform_stitch(stitched_image, processed[i], stitch_dir, spacing_width, spacing_color, custom_spacing_color)
            
            elif direction == "Grid_2x2":
                max_dim = max(min(img.shape[1], img.shape[2]) for img in present_images)
                p_img1 = get_processed_image(image1, max_dim, max_dim, "crop")
                p_img2 = get_processed_image(image2, max_dim, max_dim, "crop")
                p_img3 = get_processed_image(image3, max_dim, max_dim, "crop")
                row1 = self._perform_stitch(p_img1, p_img2, "right", spacing_width, spacing_color, custom_spacing_color)
                row2 = self._perform_stitch(p_img3, None, "right", spacing_width, spacing_color, custom_spacing_color)
                stitched_image = self._perform_stitch(row1, row2, "down", spacing_width, spacing_color, custom_spacing_color)

            else: # Compound crop
                is_vertical_first = direction.startswith("V_then_H")
                main_stitch_dir = "down" if is_vertical_first else "right"
                secondary_stitch_dir = "right" if is_vertical_first else "down"
                if direction.endswith("left"): secondary_stitch_dir = "left"
                if direction.endswith("up"): secondary_stitch_dir = "up"
                
                primary_pair = [img for img in [image1, image2] if img is not None]
                if not primary_pair:
                    if image3 is not None: 
                        img3_proc = get_processed_image(image3, min(image3.shape[1], image3.shape[2]), min(image3.shape[1], image3.shape[2]), "crop")
                        stitched_image = img3_proc
                    else: raise ValueError("No images provided for stitching.")
                
                elif image3 is None:
                    return self.stitch(main_stitch_dir, keep_proportion, pad_color, crop_position, spacing_width, spacing_color, custom_spacing_color, fill_transparent_background, transparent_fill_color, final_resize_mode, final_target_size, resample_filter, image1, image2, None)

                else:
                    max_dim_primary = max(min(img.shape[1], img.shape[2]) for img in primary_pair)
                    p_img1 = get_processed_image(image1, max_dim_primary, max_dim_primary, "crop")
                    p_img2 = get_processed_image(image2, max_dim_primary, max_dim_primary, "crop")
                    primary_stitch = self._perform_stitch(p_img1, p_img2, main_stitch_dir, spacing_width, spacing_color, custom_spacing_color)
                    target_dim_secondary = primary_stitch.shape[1] if is_vertical_first else primary_stitch.shape[2]
                    p_img3 = get_processed_image(image3, target_dim_secondary, target_dim_secondary, "crop")
                    stitched_image = self._perform_stitch(primary_stitch, p_img3, secondary_stitch_dir, spacing_width, spacing_color, custom_spacing_color)
        else: # Logic for "stretch", "resize", "pad", "pad_edge"
            if direction in ["right", "down", "left", "up"]:
                max_h = max(img.shape[1] for img in present_images)
                max_w = max(img.shape[2] for img in present_images)
                
                target_h, target_w = (max_h, max_w) if keep_proportion.startswith("pad") else (max_h if direction in ["right", "left"] else 0, max_w if direction in ["down", "up"] else 0)
                
                processed = [get_processed_image(img, target_w, target_h, keep_proportion) for img in present_images]
                stitch_dir = "right" if direction in ["right", "left"] else "down"
                if direction in ["left", "up"]: processed.reverse()
                
                stitched_image = processed[0]
                for i in range(1, len(processed)):
                    stitched_image = self._perform_stitch(stitched_image, processed[i], stitch_dir, spacing_width, spacing_color, custom_spacing_color)

            elif direction == "Grid_2x2":
                max_h = max(img.shape[1] for img in present_images)
                max_w = max(img.shape[2] for img in present_images)
                p_img1 = get_processed_image(image1, max_w, max_h, keep_proportion)
                p_img2 = get_processed_image(image2, max_w, max_h, keep_proportion)
                p_img3 = get_processed_image(image3, max_w, max_h, keep_proportion)
                row1 = self._perform_stitch(p_img1, p_img2, "right", spacing_width, spacing_color, custom_spacing_color)
                row2 = self._perform_stitch(p_img3, None, "right", spacing_width, spacing_color, custom_spacing_color)
                stitched_image = self._perform_stitch(row1, row2, "down", spacing_width, spacing_color, custom_spacing_color)

            else: # Compound layouts
                is_vertical_first = direction.startswith("V_then_H")
                main_stitch_dir, secondary_stitch_dir = ("down", "right") if is_vertical_first else ("right", "down")
                if direction.endswith("left"): secondary_stitch_dir = "left"
                if direction.endswith("up"): secondary_stitch_dir = "up"

                primary_imgs = [img for img in [image1, image2] if img is not None]
                primary_stitch = None
                if primary_imgs:
                    max_h1 = max(img.shape[1] for img in primary_imgs)
                    max_w1 = max(img.shape[2] for img in primary_imgs)
                    
                    target_h_primary, target_w_primary = (max_h1, max_w1) if keep_proportion.startswith("pad") else ((0, max_w1) if is_vertical_first else (max_h1, 0))

                    p_img1 = get_processed_image(image1, target_w_primary, target_h_primary, keep_proportion)
                    p_img2 = get_processed_image(image2, target_w_primary, target_h_primary, keep_proportion)
                    primary_stitch = self._perform_stitch(p_img1, p_img2, main_stitch_dir, spacing_width, spacing_color, custom_spacing_color)

                if primary_stitch is not None and image3 is not None:
                    target_h3, target_w3 = (primary_stitch.shape[1], 0) if is_vertical_first else (0, primary_stitch.shape[2])
                    p_img3 = get_processed_image(image3, target_w3, target_h3, keep_proportion)
                    stitched_image = self._perform_stitch(primary_stitch, p_img3, secondary_stitch_dir, spacing_width, spacing_color, custom_spacing_color)
                else:
                    stitched_image = primary_stitch if primary_stitch is not None else image3
        
        pbar.update(1) # Step 2: Stitching logic complete
        
        if stitched_image is None: raise ValueError("Stitching failed.")
        
        if fill_transparent_background and stitched_image.shape[-1] == 4:
            fill_color_rgba = self.hex_to_rgb(transparent_fill_color)
            fill_color_tensor = torch.tensor(fill_color_rgba, device=stitched_image.device, dtype=stitched_image.dtype) / 255.0
            background = torch.full_like(stitched_image, 0.0)
            background[..., :3], background[..., 3] = fill_color_tensor[:3], fill_color_tensor[3] if len(fill_color_rgba) == 4 else 1.0
            alpha, stitched_image_rgb = stitched_image[..., 3:], stitched_image[..., :3]
            blended_rgb = (stitched_image_rgb * alpha) + (background[..., :3] * (1 - alpha))
            stitched_image = torch.cat([blended_rgb, torch.ones_like(alpha)], dim=-1)
        
        pbar.update(1) # Step 3: Background fill complete
        
        # Final Resizing and Supersampling
        if final_resize_mode != "none":
            B, H, W, C = stitched_image.shape
            if W > 0 and H > 0:
                # 1. Determine final target dimensions
                if final_resize_mode == "resize_longer_side":
                    ratio = final_target_size / max(W, H)
                else: # resize_shorter_side
                    ratio = final_target_size / min(W, H)
                
                target_width = round(W * ratio)
                target_height = round(H * ratio)

                # 2. Apply supersampling if enabled
                if supersample_factor > 1.0:
                    # Upscale to supersampled dimensions first
                    ss_width = int(target_width * supersample_factor)
                    ss_height = int(target_height * supersample_factor)
                    
                    # Upscale the original stitched image to the supersampled size using the main interpolation method
                    temp_image = comfy.utils.common_upscale(stitched_image.movedim(-1,1), ss_width, ss_height, resample_filter, "disabled").movedim(1,-1)
                    
                    # Downscale to the final target size for anti-aliasing using the new downsample interpolation method
                    stitched_image = comfy.utils.common_upscale(temp_image.movedim(-1,1), target_width, target_height, final_downsample_interpolation, "disabled").movedim(1,-1)
                else:
                    # No supersampling, just resize to target
                    stitched_image = comfy.utils.common_upscale(stitched_image.movedim(-1,1), target_width, target_height, resample_filter, "disabled").movedim(1,-1)
        elif supersample_factor > 1.0: # Supersampling without final resize (just upscale)
            ss_H = int(stitched_image.shape[1] * supersample_factor)
            ss_W = int(stitched_image.shape[2] * supersample_factor)
            stitched_image = comfy.utils.common_upscale(stitched_image.movedim(-1,1), ss_W, ss_H, resample_filter, "disabled").movedim(1,-1)

        pbar.update(1) # Step 4: Final resizing/supersampling complete

        # Apply Clarity (Midtone Contrast)
        if abs(clarity_strength) > 1e-6:
            # Convert to B, C, H, W for TF.gaussian_blur
            image_bchw = stitched_image.movedim(-1, 1)
            
            clarity_blur_sigma = max(1.0, min(image_bchw.shape[2], image_bchw.shape[3]) / 50.0)
            kernel_size_clarity = self._calculate_gaussian_kernel_size(clarity_blur_sigma)
            
            blurred_image_for_clarity = TF.gaussian_blur(image_bchw, kernel_size=(kernel_size_clarity, kernel_size_clarity), sigma=(clarity_blur_sigma, clarity_blur_sigma))
            detail_for_clarity = image_bchw - blurred_image_for_clarity 
            clarity_effect_scale = 1.0 
            image_bchw = image_bchw + detail_for_clarity * clarity_strength * clarity_effect_scale
            
            stitched_image = image_bchw.clamp(0.0, 1.0).movedim(1, -1)

        pbar.update(1) # Step 5: Clarity adjustment complete

        return (stitched_image,)

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

        return (out_image, out_image.shape[2], out_image.shape[1],)
