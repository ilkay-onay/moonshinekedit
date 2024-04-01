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


import pickle
from PyQt6.QtWidgets import QFileDialog

def reset_values(main_window):
    main_window.thr_value.clear()
    main_window.thr_block_size.clear()
    main_window.thr_c.clear()
    main_window.brd_r.clear()
    main_window.brd_b.clear()
    main_window.brd_g.clear()
    main_window.brd_w.clear()
    main_window.bl_kernelsize.clear()
    main_window.gmm_value.clear()

    main_window.thr_otsu.setChecked(False)
    main_window.brd_replicate.setChecked(False)
    main_window.bl_filter.setChecked(False)
    main_window.bl_normal.setChecked(False)
    main_window.bl_bilateral.setChecked(False)
    main_window.bl_median.setChecked(False)
    main_window.bl_gaussian.setChecked(False)
    main_window.gmm_sharpining.setChecked(False)
    main_window.gmm_outline.setChecked(False)
    main_window.bda_sobel.setChecked(False)
    main_window.bda_canny.setChecked(False)
    main_window.bda_deriche.setChecked(False)
    main_window.bda_harris.setChecked(False)

    main_window.bda_laplacian.setChecked(False)
    main_window.dt_viola.setChecked(False)
    main_window.dt_kontur.setChecked(False)
    main_window.dt_watershed.setChecked(False)

def save_parameters(main_window):
    parameters = {
        "thr_value": main_window.thr_value.text(),
        "thr_block_size": main_window.thr_block_size.text(),
        "thr_c": main_window.thr_c.text(),
        "brd_r": main_window.brd_r.text(),
        "brd_b": main_window.brd_b.text(),
        "brd_g": main_window.brd_g.text(),
        "brd_w": main_window.brd_w.text(),
        "bl_kernelsize": main_window.bl_kernelsize.text(),
        "gmm_value": main_window.gmm_value.text(),
        "thr_otsu": main_window.thr_otsu.isChecked(),
        "brd_replicate": main_window.brd_replicate.isChecked(),
        "bl_filter": main_window.bl_filter.isChecked(),
        "bl_normal": main_window.bl_normal.isChecked(),
        "bl_bilateral": main_window.bl_bilateral.isChecked(),
        "bl_median": main_window.bl_median.isChecked(),
        "bl_gaussian": main_window.bl_gaussian.isChecked(),
        "gmm_sharpining": main_window.gmm_sharpining.isChecked(),
        "gmm_outline": main_window.gmm_outline.isChecked(),
        "bda_sobel": main_window.bda_sobel.isChecked(),
        "bda_canny": main_window.bda_canny.isChecked(),
        "bda_deriche": main_window.bda_deriche.isChecked(),
        "bda_harris": main_window.bda_harris.isChecked(),

        "bda_laplacian": main_window.bda_laplacian.isChecked(),
        "dt_viola": main_window.dt_viola.isChecked(),
        "dt_kontur": main_window.dt_kontur.isChecked(),
        "dt_watershed": main_window.dt_watershed.isChecked(),
    }

    filename, _ = QFileDialog.getSaveFileName(main_window, 'Save Parameters', '', 'Pickle files (*.pkl)')
    if filename:
        with open(filename, 'wb') as file:
            pickle.dump(parameters, file)

def load_parameters(main_window):
    filename, _ = QFileDialog.getOpenFileName(main_window, 'Load Parameters', '', 'Pickle files (*.pkl)')
    if filename:
        with open(filename, 'rb') as file:
            parameters = pickle.load(file)
            
            main_window.thr_value.setText(parameters.get("thr_value", ""))
            main_window.thr_block_size.setText(parameters.get("thr_block_size", ""))
            main_window.thr_c.setText(parameters.get("thr_c", ""))
            main_window.brd_r.setText(parameters.get("brd_r", ""))
            main_window.brd_b.setText(parameters.get("brd_b", ""))
            main_window.brd_g.setText(parameters.get("brd_g", ""))
            main_window.brd_w.setText(parameters.get("brd_w", ""))
            main_window.bl_kernelsize.setText(parameters.get("bl_kernelsize", ""))
            main_window.gmm_value.setText(parameters.get("gmm_value", ""))

            main_window.thr_otsu.setChecked(parameters.get("thr_otsu", False))
            main_window.brd_replicate.setChecked(parameters.get("brd_replicate", False))
            main_window.bl_filter.setChecked(parameters.get("bl_filter", False))
            main_window.bl_normal.setChecked(parameters.get("bl_normal", False))
            main_window.bl_bilateral.setChecked(parameters.get("bl_bilateral", False))
            main_window.bl_median.setChecked(parameters.get("bl_median", False))
            main_window.bl_gaussian.setChecked(parameters.get("bl_gaussian", False))
            main_window.gmm_sharpining.setChecked(parameters.get("gmm_sharpining", False))
            main_window.gmm_outline.setChecked(parameters.get("gmm_outline", False))
            main_window.bda_sobel.setChecked(parameters.get("bda_sobel", False))
            main_window.bda_canny.setChecked(parameters.get("bda_canny", False))
            main_window.bda_deriche.setChecked(parameters.get("bda_deriche", False))
            main_window.bda_harris.setChecked(parameters.get("bda_harris", False))

            main_window.bda_laplacian.setChecked(parameters.get("bda_laplacian", False))
            main_window.dt_viola.setChecked(parameters.get("dt_viola", False))
            main_window.dt_kontur.setChecked(parameters.get("dt_kontur", False))
            main_window.dt_watershed.setChecked(parameters.get("dt_watershed", False))
