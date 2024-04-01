#   MoonShineKedit, an open source Qt project designed to provide a user-friendly graphical interface for image processing tasks. 
#   Copyright (C) 2023 İlkay Onay
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QFileDialog, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import QTimer, Qt
import cv2
import numpy as np
from ui_main import Ui_MainWindow
from image_loader import ImageLoader
from exit_handler import ExitHandler
from about_dialog import show_about_dialog
from parameter_operations import save_parameters, load_parameters, reset_values
from save_error import show_non_action_warning
import os

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('icon.ico'))
        self.btn_loadimg.clicked.connect(self.load_image)
        self.btn_exit.clicked.connect(self.handle_exit)
        self.actbtn_about.triggered.connect(self.show_about_dialog)
        self.btn_reset.clicked.connect(self.reset_values)
        self.btn_savepar.clicked.connect(self.save_parameters)
        self.btn_loadpar.clicked.connect(self.load_parameters)
        self.btn_generateimg.clicked.connect(self.process_image)
        self.btn_saveimg.clicked.connect(self.save_generated_image)
        self.scene_org = QGraphicsScene()
        self.pht_org.setScene(self.scene_org)
        self.scene_new = QGraphicsScene()
        self.pgt_new.setScene(self.scene_new)
        self.image_loader = ImageLoader(self.pht_org)
        self.generated_image = None
        QTimer.singleShot(1000, self.show_about_dialog)
        self.video_capture = cv2.VideoCapture(0)  
        self.start_camera()
        self.start_realtime_processing()

    def start_camera(self):
        timer = QTimer(self)
        timer.timeout.connect(self.update_camera)
        timer.start(30)  

    def start_realtime_processing(self):
        self.otsu_timer = QTimer(self)
        self.otsu_timer.timeout.connect(self.realtime_otsu)
        self.otsu_timer.start(30)  

        self.viola_timer = QTimer(self)
        self.viola_timer.timeout.connect(self.realtime_viola)
        self.viola_timer.start(30)  

    def update_camera(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.display_in_cmr_org(pixmap)

    def display_in_cmr_org(self, pixmap):
        scene = QGraphicsScene(self)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.cmr_org.setScene(scene)
        self.cmr_org.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def realtime_otsu(self):
        if self.thr_otsu.isChecked():
            ret, frame = self.video_capture.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, otsu_frame = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                h, w = otsu_frame.shape
                bytes_per_line = 1 * w
                q_img = QImage(otsu_frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
                pixmap = QPixmap.fromImage(q_img)
                self.display_in_cmr_out(pixmap)

    def realtime_viola(self):
        if self.dt_viola.isChecked():
            ret, frame = self.video_capture.read()
            if ret:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascade_path = 'haarcascade_frontalface_default.xml'  
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if face_cascade.empty():
                    print("Error: Unable to load the Haar Cascade classifier.")
                    return
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=6)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 10)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.display_in_cmr_out(pixmap)

    def display_in_cmr_out(self, pixmap):
        scene = QGraphicsScene(self)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.cmr_out.setScene(scene)
        self.cmr_out.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    def load_image(self):
        self.image_loader.load_image()

    def show_about_dialog(self):
        show_about_dialog(self)

    def handle_exit(self):
        exit_handler = ExitHandler(self.generated_image)
        exit_handler.show_message_box()

    def reset_values(self):
        reset_values(self)

    def save_parameters(self):
        save_parameters(self)

    def load_parameters(self):
        load_parameters(self)

    def detect_faces(self):
        gray_image = cv2.cvtColor(self.generated_image, cv2.COLOR_RGBA2GRAY)
        cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("Error: Unable to load the Haar Cascade classifier.")
            return
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=6)
        for (x, y, w, h) in faces:
            cv2.rectangle(self.generated_image, (x, y), (x + w, y + h), (255, 0, 0), 10)
        self.display_image(self.generated_image, self.scene_new)

    def detect_contours(self):
        pixmap_item = self.scene_org.items()[0]
        original_pixmap = pixmap_item.pixmap()
        original_image = original_pixmap.toImage()
        width, height = original_image.width(), original_image.height()
        byte_array = original_image.bits().asstring(width * height * 4)
        numpy_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((height, width, 4))
        gray_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.generated_image, contours, -1, (0, 255, 0), 2)
        self.display_image(self.generated_image, self.scene_new)

    def process_image(self):
        pixmap_item = self.scene_org.items()[0]
        original_pixmap = pixmap_item.pixmap()
        original_image = original_pixmap.toImage()
        width, height = original_image.width(), original_image.height()
        byte_array = original_image.bits().asstring(width * height * 4)
        numpy_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((height, width, 4))

        if self.thr_otsu.isChecked():
            image = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
            _, self.generated_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            self.generated_image = numpy_array
            threshold_value_text = self.thr_value.text().strip()
            if threshold_value_text:
                image = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
                threshold_value = int(threshold_value_text)
                _, self.generated_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            elif self.thr_block_size.text() and self.thr_c.text():
                image = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
                block_size, c = int(self.thr_block_size.text().strip()), int(self.thr_c.text().strip())
                self.generated_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY, block_size, c)
        if self.dt_viola.isChecked():
            self.detect_faces()
        if self.dt_kontur.isChecked():
            self.detect_contours()
        if self.dt_watershed.isChecked():
            self.apply_watershed(numpy_array)

        if self.brd_r.text() and self.brd_g.text() and self.brd_b.text():
            border_color = (
                int(self.brd_b.text().strip()),
                int(self.brd_g.text().strip()),
                int(self.brd_r.text().strip())
            )
            self.generated_image = cv2.copyMakeBorder(
                self.generated_image,
                10, 10, 10, 10,
                cv2.BORDER_CONSTANT,
                value=border_color
            )

        blur_type = ""
        kernel_size = 0
        if self.bl_filter.isChecked():
            blur_type = 'Box'
            kernel_size = int(self.bl_kernelsize.text())
        elif self.bl_normal.isChecked():
            blur_type = 'Normal'
            kernel_size = int(self.bl_kernelsize.text())
        elif self.bl_bilateral.isChecked():
            blur_type = 'Bilateral'
        elif self.bl_median.isChecked():
            blur_type = 'Median'
            kernel_size = int(self.bl_kernelsize.text())
        elif self.bl_gaussian.isChecked():
            blur_type = 'Gaussian'
            kernel_size = int(self.bl_kernelsize.text())

        if blur_type and kernel_size > 0:
            self.generated_image = self.apply_blur(self.generated_image, kernel_size, blur_type)

        gamma_text = self.gmm_value.text().strip()
        if gamma_text:
            gamma_value = float(gamma_text)
            if gamma_value != 1.0:
                self.generated_image = self.apply_gamma_correction(self.generated_image, gamma_value)

        if self.gmm_sharpining.isChecked():
            sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            self.generated_image = cv2.filter2D(self.generated_image, -1, sharpening_kernel)

        if self.gmm_outline.isChecked():
            outline_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            outline_image = cv2.filter2D(self.generated_image, -1, outline_kernel)
            combined_image = self.generated_image.copy()
            combined_image += outline_image
            self.display_image(combined_image, self.scene_new)
            return

        border_color = (
            int(self.brd_r.text()) if self.brd_r.text() else 0,
            int(self.brd_g.text()) if self.brd_g.text() else 0,
            int(self.brd_b.text()) if self.brd_b.text() else 0
        )
        border_thickness_text = self.brd_w.text().strip()
        border_thickness = int(border_thickness_text) if border_thickness_text else 0
        brd_replicate = self.brd_replicate

        self.draw_border(border_thickness, brd_replicate)
        self.display_image(self.generated_image, self.scene_new)

        if self.bda_sobel.isChecked():
            self.apply_sobel_filter(numpy_array)

        if self.bda_canny.isChecked():
            self.apply_canny_edge_detection(numpy_array)

        if self.bda_deriche.isChecked():
            self.apply_deriche_filter(numpy_array)

        if self.bda_harris.isChecked():
            self.apply_harris_corners(numpy_array)

        if self.bda_laplacian.isChecked():
            self.apply_laplacian_filter(numpy_array)

    def draw_border(self, border_thickness, brd_replicate):
        generated_image_bgr = cv2.cvtColor(self.generated_image, cv2.COLOR_RGBA2BGR)
        if brd_replicate.isChecked():
            border_type = cv2.BORDER_REPLICATE
        else:
            border_type = cv2.BORDER_CONSTANT
        generated_image_bgr = cv2.copyMakeBorder(
            generated_image_bgr,
            border_thickness,
            border_thickness,
            border_thickness,
            border_thickness,
            borderType=border_type,
            value=None
        )
        self.generated_image = cv2.cvtColor(generated_image_bgr, cv2.COLOR_BGR2BGRA)
        self.display_image(self.generated_image, self.scene_new)

    def apply_watershed(self, numpy_array):
        gray_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
        blurred_image = cv2.medianBlur(gray_image, 3)
        _, img_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=7)
        sure_bg = cv2.dilate(img_open, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(img_open, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg, labels=5)
        markers = markers + 1
        markers[unknown == 255] = 0
        numpy_array_8uc3 = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2BGR)
        markers_32s = np.int32(markers)
        cv2.watershed(numpy_array_8uc3, markers_32s)
        contours, hierarchy = cv2.findContours(markers_32s, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(numpy_array, contours, i, (255, 0, 0), 5)
        numpy_array_with_alpha = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2BGRA)
        self.display_image(numpy_array_with_alpha, self.scene_new)

    def apply_blur(self, image, kernel_size, blur_type):
        if blur_type == 'Blur':
            return cv2.blur(image, (kernel_size, kernel_size))
        elif blur_type == 'Median':
            return cv2.medianBlur(image, kernel_size)
        elif blur_type == 'Box':
            return cv2.boxFilter(image, -1, (kernel_size, kernel_size))
        elif blur_type == 'Bilateral':
            if image.shape[2] == 4:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                filtered_image = cv2.bilateralFilter(rgb_image, 9, 75, 75)
                return cv2.cvtColor(filtered_image, cv2.COLOR_RGB2RGBA)
            else:
                return image
        elif blur_type == 'Gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'Normal':
            return cv2.blur(image, (kernel_size, kernel_size))
        else:
            return image

    def apply_gamma_correction(self, image, gamma=1.0):
        image_normalized = image / 255.0
        gamma_corrected = np.power(image_normalized, gamma)
        gamma_corrected = np.uint8(gamma_corrected * 255)
        return gamma_corrected

    def apply_sobel_filter(self, numpy_array):
        image_gray = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        sobel_colored = cv2.cvtColor(sobel_normalized, cv2.COLOR_GRAY2RGBA)
        self.display_image(sobel_colored, self.scene_new)

    def apply_canny_edge_detection(self, numpy_array):
        image_gray = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
        canny = cv2.Canny(image_gray, 100, 200)
        canny_colored = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGBA)
        self.display_image(canny_colored, self.scene_new)

    def apply_deriche_filter(self, numpy_array):
        image_gray = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
        alpha, kernel_size = 0.5, 3
        kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
        deriche_kernel_x, deriche_kernel_y = alpha * kx, alpha * ky
        deriche_x = cv2.filter2D(image_gray, -1, deriche_kernel_x)
        deriche_y = cv2.filter2D(image_gray, -1, deriche_kernel_y)
        edges = np.sqrt(np.square(deriche_x) + np.square(deriche_y))
        edges = (edges / np.max(edges)) * 255
        edges_colored = cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2RGBA)
        self.display_image(edges_colored, self.scene_new)

    def apply_harris_corners(self, numpy_array):
        image_gray = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
        image_gray = np.float32(image_gray)
        harris_corners = cv2.cornerHarris(image_gray, 2, 3, 0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        numpy_array[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255, 255]
        self.display_image(numpy_array, self.scene_new)

    def apply_laplacian_filter(self, numpy_array):
        image_gray = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2GRAY)
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        laplacian_colored = cv2.cvtColor(laplacian.astype(np.uint8), cv2.COLOR_GRAY2RGBA)
        self.display_image(laplacian_colored, self.scene_new)

    def save_generated_image(self):
        if self.generated_image is None:
            show_non_action_warning("Please generate an image first.")
            return

        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix('png')
        file_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)")

        if file_path:
            cv2.imwrite(file_path, cv2.cvtColor(self.generated_image, cv2.COLOR_RGBA2BGR))

    def display_image(self, image, scene):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_img)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.clear()
        scene.addItem(pixmap_item)
        scene.setSceneRect(pixmap_item.boundingRect())
        self.pgt_new.setScene(scene)
        self.pgt_new.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.pgt_new.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


def run():
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()
