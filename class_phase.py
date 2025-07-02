import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

#? 関数ファイルから必要な関数をインポート
from function_calculation import offset, plot_phase,approximation_phase

import streamlit as st
@st.cache_data
def make_phase_dict(img_phase_array, convolve_size_temp, z1, z2, x1, x2, x_axis, width_phase, height_phase, n_apr_pix, gaussian_additive_term, d_micro_to_pix_temp):
    #* offset
    img_phase_array_offset = offset(img_phase_array, convolve_size_temp, z1, z2, x1, x2, convolve = False)
    phase_offset_dict = {"offset": img_phase_array_offset}
    #* offset and convolve
    img_phase_array_offset_convolve = offset(img_phase_array, convolve_size_temp, z1, z2, x1, x2, convolve = True)
    phase_offset_convolve_dict = {"offset_convolve": img_phase_array_offset_convolve}
    #* aproximation_phase
    phase_apr_dict, popt_dict, popt_init_dict = approximation_phase(
        twolist_array = img_phase_array_offset_convolve,
        x_axis = x_axis,
        width_phase = width_phase,
        height_phase = height_phase,
        n_apr_pix = n_apr_pix,
        gaussian_additive_term = gaussian_additive_term,
        )

    phase_full_array_dict = phase_offset_dict | phase_offset_convolve_dict | phase_apr_dict
    phase_full_fig_dict = { key: plot_phase(value, d_micro_to_pix_temp) for key, value in phase_full_array_dict.items() }
    return phase_full_array_dict, phase_full_fig_dict, popt_dict, popt_init_dict


class PhaseAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, csv_file, d_micro_to_pix_temp, k_extract_microm, convolve_size_temp,Nx, h0,  n_apr_pix, z1, z2, x1, x2, gaussian_additive_term, debug_k_pix, microm_or_pix):
        self.path = csv_file
        self.d_micro_to_pix_temp = d_micro_to_pix_temp
        self.k_extract_microm = k_extract_microm
        self.convolve_size_temp = convolve_size_temp
        self.Nx = Nx
        self.h0 = h0
        self.n_apr_pix = n_apr_pix #* 近似に使用するフレーム数
        self.z1 = z1
        self.z2 = z2
        self.x1 = x1
        self.x2 = x2
        self.gaussian_additive_term = gaussian_additive_term
        self.debug_k_pix = debug_k_pix


        #* csvファイル読み込み
        self.df = pd.read_csv(self.path, encoding="cp932",index_col = None, header = None, engine = 'python')
        self.img_phase_array = self.df.to_numpy()

        #* 画像の横幅、縦幅の取得、x軸の作成
        self.width_phase = len(self.img_phase_array[0])
        self.height_phase = len(self.img_phase_array)
        self.x_axis = np.arange(-Nx, -Nx + self.width_phase)

        #* 位相の計算結果、近似パラメータを辞書として取得
        self.phase_full_array_dict, self.phase_full_fig_dict, self.popt_dict, self.popt_init_dict = make_phase_dict(
            img_phase_array = self.img_phase_array,
            convolve_size_temp = self.convolve_size_temp,
            z1 = self.z1,
            z2 = self.z2,
            x1 = self.x1,
            x2 = self.x2,
            x_axis = self.x_axis,
            width_phase = self.width_phase,
            height_phase = self.height_phase,
            n_apr_pix = self.n_apr_pix,
            gaussian_additive_term = self.gaussian_additive_term,
            d_micro_to_pix_temp = self.d_micro_to_pix_temp
        )

        #* microm_or_pixの選択
        #? pixを直接指定するか、μmを指定するか選択
        if microm_or_pix == "k_extract_microm[μm]":
            self.k_extract_pix_phase_from_substrate =  int(self.k_extract_microm * self.d_micro_to_pix_temp)
            self.k_extract_pix_phase_from_bottom = self.k_extract_pix_phase_from_substrate + self.h0
            self.k_extract_microm_phase_from_substrate = self.k_extract_microm
            self.k_extract_microm_phase_from_bottom =round(self.k_extract_pix_phase_from_bottom / self.d_micro_to_pix_temp, 2)
        elif microm_or_pix == "debug_k_pix[pix]":
            self.k_extract_pix_phase_from_bottom = debug_k_pix
            self.k_extract_pix_phase_from_substrate = self.k_extract_pix_phase_from_bottom - self.h0
            self.k_extract_microm_phase_from_bottom =  round(self.k_extract_pix_phase_from_bottom / self.d_micro_to_pix_temp,2)
            self.k_extract_microm_phase_from_substrate = round(self.k_extract_pix_phase_from_substrate / self.d_micro_to_pix_temp,2)
        self.k_extract_pix_phase_from_top = self.height_phase - self.k_extract_pix_phase_from_bottom




    # def make_dict(self):
    #     #* offset
    #     img_phase_array_offset = offset(self.img_phase_array, self.convolve_size_temp, self.z1, self.z2, self.x1, self.x2, convolve = False)
    #     phase_offset_dict = {"offset": img_phase_array_offset}
    #     #* offset and convolve
    #     img_phase_array_offset_convolve = offset(self.img_phase_array, self.convolve_size_temp, self.z1, self.z2, self.x1, self.x2, convolve = True)
    #     phase_offset_convolve_dict = {"offset_convolve": img_phase_array_offset_convolve}
    #     #* aproximation_phase
    #     phase_apr_dict, popt_dict, popt_init_dict = approximation_phase(
    #         twolist_array = img_phase_array_offset_convolve,
    #         x_axis = self.x_axis,
    #         width_phase = self.width_phase,
    #         height_phase = self.height_phase,
    #         n_apr_pix = self.n_apr_pix,
    #         gaussian_additive_term = self.gaussian_additive_term,
    #         )

    #     phase_full_array_dict = phase_offset_dict | phase_offset_convolve_dict | phase_apr_dict
    #     phase_full_fig_dict = { key: plot_phase(value, self.d_micro_to_pix_temp) for key, value in phase_full_array_dict.items() }
    #     return phase_full_array_dict, phase_full_fig_dict, popt_dict, popt_init_dict