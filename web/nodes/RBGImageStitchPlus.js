import { app } from "/scripts/app.js";

app.registerExtension({
    name: "RBGSuitePack.ImageStitchPlus.Icons",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RBGImageStitchPlus") {

            const WIDGETS_TO_UPDATE = ["direction", "keep_proportion", "crop_position", "spacing_color"];

            // --- OnConfigure: Apply icons to widgets ---
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) onConfigure.apply(this, arguments);

                const updateWidget = (widgetName, map, isColor = false) => {
                    const widget = this.widgets.find(w => w.name === widgetName);
                    if (!widget) return;

                    // Add a class to the widget's element for styling
                    if (widget.inputEl && widget.inputEl.parentElement) {
                        widget.inputEl.parentElement.classList.add("rbg-styled-widget");
                    }

                    // Check if the widget has already been updated
                    if (widget.options.values && typeof widget.options.values[0] === 'object') {
                        return;
                    }

                    const values = widget.options.values || [];
                    
                    widget.options.values = values.map(value => {
                        const item = map[value];
                        if (item) {
                            let content;
                            const wrapperStyle = `display: inline-block; width: calc(100% - 20px); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;`;
                            if (isColor) {
                                let style = `display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; vertical-align: middle; border: 1px solid #555;`;
                                if (value === 'custom') {
                                    style += ` background: conic-gradient(red, yellow, lime, aqua, blue, magenta, red);`;
                                } else {
                                    style += ` background-color: ${item};`;
                                }
                                content = `<span style="${wrapperStyle}"><span style="${style}"></span>${value}</span>`;
                            } else {
                                content = `<span style="${wrapperStyle}"><i class="material-icons" style="vertical-align: middle; margin-right: 5px;">${item}</i>${value}</span>`;
                            }
                            return { 
                                content: content, 
                                value: value, 
                                text: value, 
                                toString: function() { return this.text; },
                                toJSON: function() { return this.value; }
                            };
                        }
                        return value;
                    });
                };

                const keepProportionIcons = { "resize": "photo_size_select_large", "pad": "aspect_ratio", "pad_edge": "fullscreen_exit", "crop": "crop" };
                const directionIcons = { "right": "arrow_forward", "down": "arrow_downward", "left": "arrow_back", "up": "arrow_upward", "H_then_V_down": "south_east", "H_then_V_up": "north_east", "V_then_H_right": "south_east", "V_then_H_left": "south_west", "Grid_2x2": "grid_view" };
                const cropPositionIcons = { "center": "center_focus_strong", "top": "vertical_align_top", "bottom": "vertical_align_bottom", "left": "align_horizontal_left", "right": "align_horizontal_right" };
                const spacingColorMap = { "white": "#FFFFFF", "black": "#000000", "red": "#FF0000", "green": "#00FF00", "blue": "#0000FF", "custom": "custom" };

                updateWidget("keep_proportion", keepProportionIcons);
                updateWidget("direction", directionIcons);
                updateWidget("crop_position", cropPositionIcons);
                updateWidget("spacing_color", spacingColorMap, true);
            };

            // Inject CSS to ensure arrows are clickable
            const style = document.createElement('style');
            style.innerHTML = `
                .rbg-styled-widget .combo-arrow {
                    pointer-events: auto !important;
                    z-index: 100 !important;
                }
            `;
            document.head.appendChild(style);
        }
    },
});
