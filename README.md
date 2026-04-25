```markdown
# MoonShineKedit

## Overview

MoonShineKedit is a sophisticated, open-source image processing application built with PyQt6. It provides a user-friendly graphical interface for a wide array of image manipulation tasks, from basic loading and saving to advanced filtering, thresholding, and edge detection algorithms. The application leverages the power of OpenCV for its core image processing capabilities, offering real-time previews of applied filters and effects. MoonShineKedit is designed to be both a powerful tool for image manipulation and an educational platform for understanding various image processing techniques.

## Features

*   **Image Loading and Display:** Load images in various formats (PNG, JPG, BMP, GIF, TIFF) and display them in dedicated views for original and processed images.
*   **Real-time Camera Feed:** Integrates with the system's camera to provide a live video stream, allowing for real-time application of filters.
*   **Extensive Filtering Options:**
    *   **Thresholding:** Simple, Adaptive, and Otsu's thresholding methods.
    *   **Blurring:** Box, Normal, Bilateral, Median, and Gaussian blur filters with adjustable kernel sizes.
    *   **Gamma Correction:** Adjust image brightness and contrast using gamma correction.
    *   **Sharpening & Outlining:** Apply sharpening kernels and outline effects.
*   **Advanced Image Processing:**
    *   **Border Detection Algorithms:** Sobel, Laplacian, Canny, Deriche, and Harris corner detection.
    *   **Contour Detection:** Identify and draw contours within an image.
    *   **Face Detection:** Utilizes Haar Cascades for real-time and on-demand face detection.
    *   **Watershed Algorithm:** Advanced segmentation technique for image analysis.
*   **Parameter Management:** Save and load custom parameter configurations for reproducible results.
*   **User-Friendly Interface:** Intuitive layout with clear labeling and organized controls.
*   **About and License Information:** Provides details about the software, its author, and the GNU GPL v3.0 license.

## Project Structure

```
├── LICENSE
├── LICENSE.txt
├── about_dialog.py
├── exit_handler.py
├── haarcascade_frontalface_default.xml
├── image_loader.py
├── main.py
├── main.ui
├── parameter_operations.py
├── save_error.py
├── ui_main.py
```

## Getting Started

To build and run MoonShineKedit, you will need Python and the PyQt6 library installed.

1.  **Install Dependencies:**
    ```bash
    pip install PyQt6 opencv-python numpy
    ```

2.  **Run the Application:**
    Navigate to the project directory in your terminal and execute the main Python script:
    ```bash
    python main.py
    ```

## License

This project is licensed under the **GNU General Public License v3.0**.

```
Copyright (C) 2023 İlkay Onay

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
```