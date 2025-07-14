import { app } from "/scripts/app.js";

app.registerExtension({
    name: "RBGSuitePack.PadPro.Tooltips",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RBGPadPro") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const tooltips = {
                    pad_mode: "Choose how to handle the padding. 'Pad' adds space, 'pad_edge' uses edge colors, 'transparent_fill' fills transparent areas, and 'crop' cuts the image.",
                    pad_left: "Amount of padding to add to the left side.",
                    pad_right: "Amount of padding to add to the right side.",
                    pad_top: "Amount of padding to add to the top.",
                    pad_bottom: "Amount of padding to add to the bottom.",
                    pad_feathering: "Blends the image edges with the background for a smoother transition.",
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
                    resample_filter: "The interpolation method to use for resizing."
                };

                for (const widget of this.widgets) {
                    if (tooltips[widget.name] && widget.canvas) {
                        widget.canvas.title = tooltips[widget.name];
                    }
                }
            };
        }
    }
});
