import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "RBGSuitePack.PadPro.Complete",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RBGPadPro") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const MIN_NODE_HEIGHT_WITH_PREVIEW = 400;
            const MIN_NODE_HEIGHT_WITHOUT_PREVIEW = 250;

            const img = new Image();
            img.onload = () => node.setDirtyCanvas(true);

            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // Tooltips configuration
                const tooltips = {
                    pad_mode: "Choose how to handle the padding. 'pad' adds solid color, 'pad_edge' averages border colors, 'pad_edge_pixel' extends border pixels, and 'transparent_fill' fills alpha channels.",
                    pad_left: "Amount of padding to add to the left side.",
                    pad_right: "Amount of padding to add to the right side.",
                    pad_top: "Amount of padding to add to the top.",
                    pad_bottom: "Amount of padding to add to the bottom.",
                    mask_blur_sigma: "Applies a Gaussian blur to the generated mask for smoother inpainting seams.",
                    pad_color: "The solid color of the padded area.",
                    image_position: "The base alignment/anchor of the image on the canvas.",
                    image_offset_x: "Fine-tune the horizontal position in pixels.",
                    image_offset_y: "Fine-tune the vertical position in pixels.",
                    image_scale: "Scale the image before padding without altering the canvas bounds.",
                    fill_transparent_background: "Fill transparent areas with a solid color.",
                    transparent_fill_color: "The color to use for transparent background fill.",
                    pad_aspect_ratio: "Target aspect ratio preset.",
                    resize_mode: "Choose how to resize the final image ('resize_longer_side' or 'resize_shorter_side').",
                    target_size: "The target size in pixels for the selected side.",
                    resample_filter: "The interpolation method to use for resizing.",
                    auto_crop_aspect_ratio: "Crops the image to match the selected pad_aspect_ratio instead of expanding the canvas with padding.",
                    invert_mask: "Inverts the output mask (Standard for Outpainting).",
                    flip_horizontal: "Horizontally flips the image and mask.",
                    image_rotation: "Rotates the input image by degrees.",
                    BorderCrop_threshold: "Automatically trims solid color borders before processing.",
                    pad_noise_strength: "Strength of noise added to the padded region to prep diffusion latents.",
                    noise_seed: "Seed for procedural noise generation.",
                    noise_mode: "Color, Greyscale, or Greyscale+Mask noise.",
                    divisibility: "Enforces dimensions to be multiples of 8, 16, 32, 64, etc. for model compatibility."
                };

                for (const widget of this.widgets) {
                    if (tooltips[widget.name] && widget.canvas) {
                        widget.canvas.title = tooltips[widget.name];
                    }
                }

                // Output info widget
                const textWidget = this.addWidget("text", "output_text", "", {});
                textWidget.serialize = false;

                // Reset Button
                const resetButton = this.addWidget("button", "Reset to Defaults 🗑️", "reset", () => {
                    const defaults = nodeData.input.required;
                    for (const widget of this.widgets) {
                        if (defaults[widget.name]) {
                            const defaultValue = defaults[widget.name][1]?.default ?? defaults[widget.name][0];
                            widget.value = defaultValue;
                        }
                    }
                });
                resetButton.serialize = false;

                // Preview handling
                const previewWidget = this.widgets.find(w => w.name === "show_preview");
                if (previewWidget) {
                    const originalCallback = previewWidget.callback;
                    previewWidget.callback = (value) => {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        this.size[1] = value ? MIN_NODE_HEIGHT_WITH_PREVIEW : MIN_NODE_HEIGHT_WITHOUT_PREVIEW;
                        this.setDirtyCanvas(true, true);
                    };
                }

                this.size[1] = previewWidget?.value ? MIN_NODE_HEIGHT_WITH_PREVIEW : MIN_NODE_HEIGHT_WITHOUT_PREVIEW;

                api.addEventListener("rbg_pad_pro_preview", (event) => {
                    if (event.detail.image) {
                        img.src = event.detail.image;
                    }
                });
            };

            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                onDrawBackground?.apply(this, arguments);

                if (this.flags.collapsed) return;

                const showPreview = this.widgets.find(w => w.name === "show_preview")?.value;
                if (!showPreview || !img.src) return;

                const [w, h] = this.size;
                const lastWidget = this.widgets[this.widgets.length - 1];
                const lastWidgetY = lastWidget?.last_y || 0;
                const PADDING = 10;
                const IMAGE_Y_OFFSET = lastWidgetY + 30;
                const imageAreaHeight = h - IMAGE_Y_OFFSET - PADDING;

                if (img.src && imageAreaHeight > 50) {
                    const aspectRatio = img.width / img.height;
                    let drawWidth = w - 2 * PADDING;
                    let drawHeight = drawWidth / aspectRatio;

                    if (drawHeight > imageAreaHeight) {
                        drawHeight = imageAreaHeight;
                        drawWidth = drawHeight * aspectRatio;
                    }
                    const x = PADDING + (w - 2 * PADDING - drawWidth) / 2;
                    ctx.drawImage(img, x, IMAGE_Y_OFFSET, drawWidth, drawHeight);
                }
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (message?.text) {
                    const widget = this.widgets.find(w => w.name === "output_text");
                    if (widget) {
                        widget.value = message.text[0];
                    }
                }
            };
        }
    }
});