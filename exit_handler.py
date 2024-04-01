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
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from PyQt6.QtGui import QIcon
import cv2
from save_error import NoResultsError, show_non_action_warning  # Import the necessary items

class ExitHandler:
    def __init__(self, generated_image):
        self.generated_image = generated_image

    def show_message_box(self):
        msg_box = QMessageBox()
        msg_box.setWindowIcon(QIcon("icon.ico"))
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText("Are you sure you want to exit this program?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        msg_box.setStyleSheet("QMessageBox { background-color: #333; color: white; } QLabel { color: white; } QPushButton { background-color: #555; color: white; }")

        save_and_exit_button = msg_box.addButton("Save and Exit", QMessageBox.ButtonRole.ActionRole)
        msg_box.setDefaultButton(save_and_exit_button)

        result = msg_box.exec()

        if msg_box.clickedButton() == save_and_exit_button:
            self.save_variables()
            try:
                self.save_to_file()
            except NoResultsError:
                show_non_action_warning()
            sys.exit()
        elif result == QMessageBox.StandardButton.Yes:
            sys.exit()
            
    def save_to_file(self):
        if self.generated_image is not None:
            file_dialog = QFileDialog()
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tiff)")
            file_dialog.setWindowTitle("Save Image File")

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                file_path = file_dialog.selectedFiles()[0]
                cv2.imwrite(file_path, self.generated_image)
        else:
            raise NoResultsError("No generated image to save")