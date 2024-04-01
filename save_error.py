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

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QIcon

class NoResultsError(Exception):
    pass

def show_non_action_warning():
    msg = QMessageBox()
    msg.setStyleSheet("QMessageBox { background-color: #333; color: white; } QLabel { color: white; } QPushButton { background-color: #555; color: white; }")
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setText("No Results Error")
    msg.setInformativeText("There are no results to save. Please do the calculations with the tool first.")
    msg.setWindowTitle("Warning")
    msg.setWindowIcon(QIcon("icon.ico"))
    msg.exec()
