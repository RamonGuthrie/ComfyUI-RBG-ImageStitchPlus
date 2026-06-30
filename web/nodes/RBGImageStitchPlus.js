import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

// Function to load Material Icons
const loadMaterialIcons = () => {
    const linkId = 'material-icons-font';
    if (!document.getElementById(linkId)) {
        const link = document.createElement('link');
        link.id = linkId;
        link.rel = 'stylesheet';
        link.href = 'https://fonts.googleapis.com/icon?family=Material+Icons';
        document.head.appendChild(link);
    }
};

// Color utility functions
const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
};

const rgbToHex = (r, g, b) => {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
};

const rgbToHsl = (r, g, b) => {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
        h = s = 0; // achromatic
    } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return {h: h * 360, s: s * 100, l: l * 100};
};

// Create advanced color picker widget
const createAdvancedColorPicker = (node, inputName, inputData, app) => {
    const container = document.createElement("div");
    container.style.cssText = `
        background: #2a2a2a;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
        border: 1px solid #444;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    `;

    let currentColor = inputData[1]?.default || "#ffffff";
    
    // Header with label
    const header = document.createElement("div");
    header.innerHTML = `<span style="color: #ccc; font-size: 12px; font-weight: 500;">${inputName}</span>`;
    container.appendChild(header);

    // Color preview and hex input row
    const colorRow = document.createElement("div");
    colorRow.style.cssText = "display: flex; align-items: center; gap: 8px; margin: 8px 0;";
    
    const colorPreview = document.createElement("div");
    colorPreview.style.cssText = `
        width: 32px;
        height: 24px;
        border-radius: 6px;
        background: ${currentColor};
        border: 1px solid #555;
        cursor: pointer;
        flex-shrink: 0;
    `;
    
    const hexInput = document.createElement("input");
    hexInput.type = "text";
    hexInput.value = currentColor.toUpperCase();
    hexInput.style.cssText = `
        background: #1a1a1a;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 6px 8px;
        color: #fff;
        font-size: 12px;
        font-family: monospace;
        flex: 1;
        min-width: 80px;
    `;
    
    const formatSelect = document.createElement("select");
    formatSelect.innerHTML = `
        <option value="hex">Hex</option>
        <option value="rgb">RGB</option>
        <option value="hsl">HSL</option>
    `;
    formatSelect.style.cssText = `
        background: #1a1a1a;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 6px;
        color: #fff;
        font-size: 11px;
    `;

    colorRow.appendChild(colorPreview);
    colorRow.appendChild(hexInput);
    colorRow.appendChild(formatSelect);
    container.appendChild(colorRow);

    // Saturation display
    const saturationRow = document.createElement("div");
    saturationRow.style.cssText = "display: flex; justify-content: space-between; align-items: center; margin: 4px 0;";
    
    const saturationLabel = document.createElement("span");
    saturationLabel.style.cssText = "color: #999; font-size: 11px;";
    saturationLabel.textContent = "Saturation";
    
    const saturationValue = document.createElement("span");
    saturationValue.style.cssText = "color: #ccc; font-size: 11px; font-weight: 500;";
    
    saturationRow.appendChild(saturationLabel);
    saturationRow.appendChild(saturationValue);
    container.appendChild(saturationRow);

    // Update saturation display
    const updateSaturation = (color) => {
        const rgb = hexToRgb(color);
        if (rgb) {
            const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
            saturationValue.textContent = Math.round(hsl.s) + "%";
        }
    };

    // Update color function
    const updateColor = (newColor) => {
        currentColor = newColor;
        colorPreview.style.background = currentColor;
        
        const format = formatSelect.value;
        if (format === "hex") {
            hexInput.value = currentColor.toUpperCase();
        } else if (format === "rgb") {
            const rgb = hexToRgb(currentColor);
            if (rgb) hexInput.value = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
        } else if (format === "hsl") {
            const rgb = hexToRgb(currentColor);
            if (rgb) {
                const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
                hexInput.value = `hsl(${Math.round(hsl.h)}, ${Math.round(hsl.s)}%, ${Math.round(hsl.l)}%)`;
            }
        }
        
        updateSaturation(currentColor);
        
        // Update the widget value
        if (widget.callback) {
            widget.callback(currentColor);
        }
    };

    // Color picker click handler
    colorPreview.addEventListener("click", () => {
        const input = document.createElement("input");
        input.type = "color";
        input.value = currentColor;
        input.style.position = "absolute";
        input.style.left = "-9999px";
        document.body.appendChild(input);
        
        input.addEventListener("change", () => {
            updateColor(input.value);
            document.body.removeChild(input);
        });
        
        input.addEventListener("blur", () => {
            if (document.body.contains(input)) {
                document.body.removeChild(input);
            }
        });
        
        input.click();
    });

    // Hex input handler
    hexInput.addEventListener("input", () => {
        let value = hexInput.value;
        if (formatSelect.value === "hex") {
            if (value.match(/^#?[0-9a-fA-F]{0,6}$/)) {
                if (!value.startsWith("#")) value = "#" + value;
                if (value.length === 7) {
                    updateColor(value);
                }
            }
        }
    });

    // Format change handler
    formatSelect.addEventListener("change", () => {
        updateColor(currentColor);
    });

    // Initial saturation update
    updateSaturation(currentColor);

    // Create widget wrapper
    const widget = {
        type: "COLOR_ADVANCED",
        name: inputName,
        value: currentColor,
        element: container,
        callback: null,
        
        // Make it compatible with ComfyUI's widget system
        computeSize: () => [200, 85],
        
        draw: function(ctx, node, width, y) {
            // This widget uses DOM elements instead of canvas drawing
            return 85; // Return height
        },
        
        mouse: function(event, pos, node) {
            // Handle mouse events if needed
            return false;
        }
    };

    // Set up callback to update node
    widget.callback = (value) => {
        widget.value = value;
        currentColor = value;
        if (node.onWidgetChanged) {
            node.onWidgetChanged(widget.name, value);
        }
    };

    return { widget, element: container };
};

app.registerExtension({
    name: "RBGSuitePack.ImageStitchPlus.ModernColorPicker",
    
    init() {
        // Register the advanced color widget
        if (ComfyWidgets && !ComfyWidgets.COLOR_ADVANCED) {
            ComfyWidgets.COLOR_ADVANCED = createAdvancedColorPicker;
        }
    },
    
    getWidgetTypes() {
        return {
            COLOR: createAdvancedColorPicker
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RBGImageStitchPlus") {
            loadMaterialIcons();
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Replace COLOR widgets with advanced ones
                setTimeout(() => {
                    if (this.widgets) {
                        this.widgets.forEach((widget, index) => {
                            if (widget.type === "COLOR" || (widget.name && widget.name.includes("color"))) {
                                const inputData = nodeData.input?.required?.[widget.name] || nodeData.input?.optional?.[widget.name];
                                if (inputData && inputData[0] === "COLOR") {
                                    const { widget: newWidget, element } = createAdvancedColorPicker(this, widget.name, inputData, app);
                                    
                                    // Replace the old widget
                                    this.widgets[index] = newWidget;
                                    
                                    // Add the DOM element to the node
                                    if (!this.colorPickerElements) this.colorPickerElements = [];
                                    this.colorPickerElements.push(element);
                                    
                                    // Append to the UI (you may need to adjust this based on your UI structure)
                                    if (this.element) {
                                        this.element.appendChild(element);
                                    }
                                }
                            }
                        });
                    }
                }, 100);
                
                return r;
            };

            // Keep your existing icon logic
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) onConfigure.apply(this, arguments);

                const updateWidget = (widgetName, map, isColor = false) => {
                    const widget = this.widgets.find(w => w.name === widgetName);
                    if (!widget || widget.type === "COLOR_ADVANCED") return;

                    if (widget.options.values && typeof widget.options.values[0] === 'object') {
                        return;
                    }

                    const values = widget.options.values || [];
                    
                    widget.options.values = values.map(value => {
                        const item = map[value];
                        if (item) {
                            let content;
                            if (isColor) {
                                let style = `display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; vertical-align: middle; border: 1px solid #555;`;
                                if (value === 'custom') {
                                    style += ` background: conic-gradient(red, yellow, lime, aqua, blue, magenta, red);`;
                                } else {
                                    style += ` background-color: ${item};`;
                                }
                                content = `<span style="${style}"></span>${value}`;
                            } else {
                                content = `<i class="material-icons" style="vertical-align: middle; margin-right: 5px;">${item}</i>${value}`;
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
        }
    },
});