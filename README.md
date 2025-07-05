# ComfyUI-RBG-ImageStitchPlus 🧩

## An Advanced Image Stitching Node for ComfyUI ✨

This project provides a powerful and flexible image stitching node for ComfyUI, designed to seamlessly combine multiple images into a single canvas. Whether you're creating panoramas, grids, or complex layouts, the RBG Image Stitch Plus node offers the tools you need to get the job done.

---
![Screenshot](https://github.com/user-attachments/assets/9bc5e84f-4d33-46d5-b5c1-0fc21ef7699d)
## Features 🚀

-   **Advanced Stitching Directions:** Combine up to three images with multiple layout options:
    -   **Simple:** `right`, `down`, `left`, `up`.
    -   **Compound:** `H_then_V_down`, `H_then_V_up`, `V_then_H_right`, `V_then_H_left`.
    -   **Grid:** `Grid_2x2` for a four-quadrant layout.
-   **Intelligent Proportion Control:** Choose how to handle images of different sizes:
    -   `resize`: Resizes images to match the dimensions required by the layout.
    -   `pad`: Adds padding to smaller images to match the largest image in the relevant dimension.
    -   `pad_edge`: A unique padding mode that analyses the border pixels of an image and uses their average colour for a seamless extension.
    -   `crop`: Crops images to a uniform size. You can control the crop's origin with `crop_position` (`centre`, `top`, `left`, etc.) for precise framing.
-   **Customizable Spacing:** Add a visual separator between stitched images with a custom width and colour.
-   **Background Fill:** Fill transparent areas of the final canvas with a solid colour, perfect for ensuring consistency in your final output.
-   **Advanced Resizing & Anti-Aliasing:**
    -   **Final Resize:** Scale the entire stitched canvas based on its `longer_side` or `shorter_side` to fit specific dimensions.
    -   **Supersampling:** Apply high-quality anti-aliasing by rendering the image at a higher resolution (`supersample_factor`) and then downscaling it.
    -   **Interpolation Control:** Select specific resampling filters (e.g., `lanczos`, `bicubic`, `area`) for both resizing and the final downsample, giving you full control over the final texture and sharpness.
-   **Clarity Adjustment:** Enhance or soften the midtone contrast of the final image. This powerful feature can make an image "pop" with punchy detail or give it a soft, dreamlike feel by adjusting the `clarity_strength`.

## Experience the Power – Watch the Feature Showcase 📺 
https://github.com/user-attachments/assets/52eec166-9c79-4583-9c89-d83c2dcbe986

---

## Installation 🛠️

1.  Clone this repository into your `ComfyUI/custom_nodes` directory:
    ```bash
    git clone https://github.com/RamonGuthrie/ComfyUI-RBG-ImageStitchPlus.git
    ```
2.  Install the required dependencies by running the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```
3. Restart ComfyUI.

---

## Usage 🚀

After installation, you can find the `RBG Image Stitch Plus` node under the `RBG/ImageStitchPlus` category in ComfyUI. Connect up to three images and configure the settings to create your desired composition.

---

## Contributing ❤️

Contributions are always welcome! If you have any suggestions, improvements, or new ideas, please feel free to submit a pull request or open an issue.

---

## License 📜

This project is licensed under the MIT License.
