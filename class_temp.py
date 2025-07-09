import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

#? 関数ファイルから必要な関数をインポート
from function_calculation import refractive, solve_T, calc_temp, plot_temp

import streamlit as st
# @st.cache_data
def make_temp_dict(phase_dict, x_axis, d_micro_to_pix_temp, k_extract_microm_phase, convolve_size_temp, n_room,  Nx, Nz, l, n_apr_pix, lamda, T_room):
    #* temperature extraction
    T_dict = {}
    r_dict = {}
    for key, value in phase_dict.items():
        res1, res2, x_axis_pix_half = calc_temp(value,x_axis, Nx, Nz, 0, l, d_micro_to_pix_temp, n_room, lamda)
        T_dict[key] = res1
        r_dict[key] = res2

    T_fig_dict = { key: plot_temp(value, d_micro_to_pix_temp) for key, value in T_dict.items()}
    return T_dict, T_fig_dict, r_dict, x_axis_pix_half

class TempAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, phase_dict, x_axis,  d_micro_to_pix_temp, k_extract_microm_phase, convolve_size_temp, Nx, Nz, l, n_apr_pix, lamda, T_room):
        self.phase_dict = phase_dict
        self.x_axis = x_axis
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

        self.T_dict, self.T_fig_dict, self.r_dict, self.x_axis_pix_half = make_temp_dict(
            phase_dict = self.phase_dict,
            x_axis = self.x_axis,
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
