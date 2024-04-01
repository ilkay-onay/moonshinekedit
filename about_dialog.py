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

from PyQt6.QtWidgets import QMessageBox, QPushButton, QDialog, QVBoxLayout, QScrollArea, QTextEdit
from PyQt6.QtGui import QIcon

def show_about_dialog(parent):
    aboutbox = QMessageBox()
    aboutbox.setWindowIcon(QIcon("icon.ico"))
    aboutbox.setStyleSheet("QMessageBox { background-color: #333; color: white; } QLabel { color: white; } QPushButton { background-color: #555; color: white; }")

    aboutbox.setWindowTitle("MoonShineKedit Version 1.00")
    
    text = (
        "<b>About MoonShineKedit</b><br><br>"
        "MoonShineKedit <i>Copyright (C) 2023 İlkay Onay</i><br><br>"
        "An open source Qt project designed to provide a user-friendly graphical interface for image processing tasks.<br><br><br>"
        "This program comes with ABSOLUTELY NO WARRANTY; for details click the 'License' button.<br>"
        "This is free software, and you are welcome to redistribute it under certain conditions; click the 'License' button for details.</a>"
    )
    aboutbox.setText(text)
    license_button = QPushButton("License")
    aboutbox.addButton(license_button, QMessageBox.ButtonRole.ActionRole)
    aboutbox.addButton(QMessageBox.StandardButton.Ok)
    result = aboutbox.exec()
    if result == QMessageBox.StandardButton.Ok:
        pass
    elif aboutbox.clickedButton() == license_button:
        show_license_dialog()

def show_license_dialog():
    license_dialog = QDialog()
    license_dialog.setWindowTitle("GNU General Public License")
    license_dialog.setStyleSheet("background-color: #333; color: white;")
    license_dialog.setWindowIcon(QIcon("icon.ico"))

    scroll_area = QScrollArea(license_dialog)
    scroll_area.setWidgetResizable(True)

    license_text = QTextEdit()
    license_text.setReadOnly(True)
    
    try:
        with open("LICENSE.txt", "r", encoding="utf-8") as file:
            license_text_content = file.read()
    except FileNotFoundError:
        license_text_content = "License text not found."

    license_text.setPlainText(license_text_content)

    close_button = QPushButton("Close")
    close_button.setStyleSheet("background-color: #555; color: white;")
    close_button.clicked.connect(license_dialog.close)

    layout = QVBoxLayout()
    layout.addWidget(scroll_area)
    layout.addWidget(close_button)
    scroll_area.setWidget(license_text)

    license_dialog.setLayout(layout)
    license_dialog.setFixedSize(600, 400)

    license_dialog.exec()