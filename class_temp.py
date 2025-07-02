import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

#? 関数ファイルから必要な関数をインポート
from function_calculation import refractive, solve_T, calc_temp, plot_temp

import streamlit as st
# @st.cache_data
def make_temp_dict(phase_dict, d_micro_to_pix_temp, k_extract_microm_phase, convolve_size_temp, n_room,  Nx, Nz, l, n_apr_pix, lamda, T_room):
    #* temperature extraction
    T_dict = {}
    r_dict = {}
    for key, value in phase_dict.items():
        res1, res2 = calc_temp(value, Nx, Nz, 0, l, d_micro_to_pix_temp, n_room, lamda)
        T_dict[key] = res1
        r_dict[key] = res2

    T_fig_dict = { key: plot_temp(value, d_micro_to_pix_temp) for key, value in T_dict.items()}
    return T_dict, T_fig_dict, r_dict

class TempAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, phase_dict, d_micro_to_pix_temp, k_extract_microm_phase, convolve_size_temp, Nx, Nz, l, n_apr_pix, lamda, T_room):
        self.phase_dict = phase_dict
        self.d_micro_to_pix_temp = d_micro_to_pix_temp
        self.k_extract_microm_phase = k_extract_microm_phase
        self.convolve_size_temp = convolve_size_temp
        self.Nx = Nx
        self.Nz = Nz
        self.l = l
        self.n_apr_pix = n_apr_pix #* 近似に使用するフレーム数
        self.lamda = lamda
        self.T_room = T_room

        self.n_room = refractive(self.T_room) #*水の常温=室温として，室温における水の屈折率を算出

        self.T_dict, self.T_fig_dict, self.r_dict = make_temp_dict(
            phase_dict = self.phase_dict,
            d_micro_to_pix_temp = self.d_micro_to_pix_temp,
            k_extract_microm_phase = self.k_extract_microm_phase,
            convolve_size_temp = self.convolve_size_temp,
            n_room = self.n_room,
            Nx = self.Nx,
            Nz = self.Nz,
            l = self.l,
            n_apr_pix = self.n_apr_pix,
            lamda = self.lamda,
            T_room = self.T_room
        )

        # #* temperature extraction
        # self.T_solution_offset, self.r_offset = calc_temp(
        #     twolist_array = self.phase_dict['offset'],
        #     Nx = self.Nx,
        #     Nz = self.Nz,
        #     mode = 1,
        #     l = self.l,
        #     d_temp = self.d_micro_to_pix_temp,
        #     n_room  = self.n_room,
        #     lamda = self.lamda
        #     )
        # self.fig_offset, self.ax_offset = plot_temp(self.T_solution_offset, self.d_micro_to_pix_temp)

        # self.T_solution_offset_convolve, self.r_offset_convolve = calc_temp(
        #     twolist_array = self.phase_dict['offset_convolve'],
        #     Nx = self.Nx,
        #     Nz = self.Nz,
        #     mode = 1,
        #     l = self.l,
        #     d_temp = self.d_micro_to_pix_temp,
        #     n_room  = self.n_room,
        #     lamda = self.lamda
        #     )
        # # self.fig_offset_convolve, self.ax_offset_convolve = plot_temp(self.T_solution_offset_convolve, self.d_micro_to_pix_temp)

        # self.T_dict = {
        #         "offset": self.T_solution_offset,
        #         "offset_convolve": self.T_solution_offset_convolve
        #     }
        # self.T_fig_dict = { key: plot_temp(value, d_micro_to_pix_temp) for key, value in self.T_dict.items()}
        # self.r_dict = {
        #         "offset": self.r_offset,
        #         "offset_convolve": self.r_offset_convolve
        #     }