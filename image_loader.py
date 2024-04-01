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

from PyQt6.QtWidgets import QFileDialog, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class ImageLoader:
    def __init__(self, graphics_view):
        self.graphics_view = graphics_view

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.gif *.tiff)")
        file_dialog.setWindowTitle("Open Image File")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_path)
            self.display_image(pixmap)

    def display_image(self, pixmap):
        scene = self.graphics_view.scene()
        scene.clear()
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        self.graphics_view.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)