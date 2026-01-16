import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "RBGSuitePack.PadPro.Complete", // Updated name to ensure cache is cleared
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RBGPadPro") {
            // This function runs once when the node is created.
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const MIN_NODE_HEIGHT_WITH_PREVIEW = 400;
            const MIN_NODE_HEIGHT_WITHOUT_PREVIEW = 250;

            const img = new Image();
            img.onload = () => node.setDirtyCanvas(true);

            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // --- START: RE-ADD TOOLTIPS ---
                const tooltips = {
                    pad_mode: "Choose how to handle the padding. 'Pad' adds space, 'pad_edge' uses edge colors, 'pad_edge_pixel' extends edge pixels, 'pad_mirror' reflects the image edges, 'transparent_fill' fills transparent areas, and 'crop' cuts the image.",
                    pad_left: "Amount of padding to add to the left side.",
                    pad_right: "Amount of padding to add to the right side.",
                    pad_top: "Amount of padding to add to the top.",
                    pad_bottom: "Amount of padding to add to the bottom.",
                    mask_blur_sigma: "Applies a Gaussian blur to the generated mask, blending the padded area more smoothly.",
                    pad_color: "The color of the padded area.",
                    image_position: "The base position of the image on the canvas.",
                    image_offset_x: "Fine-tune the image's horizontal position.",
                    image_offset_y: "Fine-tune the image's vertical position.",
                    image_scale: "Scale the image before padding. The canvas size is not affected.",
                    fill_transparent_background: "Fill transparent areas with a solid color.",
                    transparent_fill_color: "The color to use for transparent areas.",
                    pad_aspect_ratio: "Automatically adjust padding to match a specific aspect ratio.",
                    resize_mode: "Choose how to resize the final image. 'resize_longer_side' and 'resize_shorter_side' maintain the aspect ratio.",
                    target_size: "The target size in pixels for the selected side.",
                    resample_filter: "The interpolation method to use for resizing.",
                    auto_crop: "Automatically crop the image to a 1:1 square aspect ratio before any other processing.",
                    invert_mask: "Invert the generated mask. Useful for outpainting workflows where you want to mask the padded area instead of the original image."
                };

                for (const widget of this.widgets) {
                    if (tooltips[widget.name] && widget.canvas) {
                        widget.canvas.title = tooltips[widget.name];
                    }
                }
                // --- END: RE-ADD TOOLTIPS ---

                // Create a dedicated text widget for the output, same as KJNodes.
                const textWidget = this.addWidget("text", "output_text", "", {});
                textWidget.serialize = false; // Don't save this text in the workflow file

                // --- START: ADD RESET BUTTON ---
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
                // --- END: ADD RESET BUTTON ---

                // --- START: ADD PREVIEW ---
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

                // Set initial size
                this.size[1] = previewWidget?.value ? MIN_NODE_HEIGHT_WITH_PREVIEW : MIN_NODE_HEIGHT_WITHOUT_PREVIEW;

                api.addEventListener("rbg_pad_pro_preview", (event) => {
                    if (event.detail.image) {
                        img.src = event.detail.image;
                    }
                });
                // --- END: ADD PREVIEW ---
            };

            // --- START: ADD PREVIEW DRAWING ---
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                onDrawBackground?.apply(this, arguments);

                if (this.flags.collapsed) return;

                const showPreview = this.widgets.find(w => w.name === "show_preview")?.value;
                if (!showPreview || !img.src) return;

                const [w, h] = this.size;
                const lastWidget = this.widgets[this.widgets.length - 1];
                const lastWidgetY = lastWidget.last_y || 0;
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

            // This function runs every time after the node is executed.
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                // Find our output widget and update its value with the message from Python.
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
