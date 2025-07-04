# ComfyUI-RBG-ImageStitchPlus 🧩

## An Advanced Image Stitching Node for ComfyUI ✨

This project provides a powerful and flexible image stitching node for ComfyUI, designed to seamlessly combine multiple images into a single canvas. Whether you're creating panoramas, grids, or complex layouts, the RBG Image Stitch Plus node offers the tools you need to get the job done.

---
![Screenshot](https://github.com/user-attachments/assets/9bc5e84f-4d33-46d5-b5c1-0fc21ef7699d)
## Features 🚀

-   **Multiple Stitching Directions:** Stitch images horizontally, vertically, in compound sequences (e.g., H then V), or in a 2x2 grid.
-   **Proportion Control:** Choose how to handle images of different sizes:
    -   `resize`: Resizes images to match.
    -   `pad`: Adds padding to smaller images to match the largest.
    -   `pad_edge`: Pads with the average color of the image's edge.
    -   `crop`: Crops images to a uniform size before stitching.
-   **Customizable Spacing:** Add spacing between stitched images with a custom width and color.
-   **Background Fill:** Fill transparent areas of the final image with a solid color.
-   **Final Resizing & Anti-Aliasing:**
    -   Resize the final stitched image based on its longer or shorter side.
    -   Apply supersampling for high-quality anti-aliasing.
-   **Clarity Adjustment:** Enhance or soften the midtone contrast of the final image for a punchier or more dreamlike look.



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
