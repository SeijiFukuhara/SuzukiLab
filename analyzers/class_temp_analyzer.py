import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from scipy.ndimage import uniform_filter1d

#? 関数ファイルから必要な関数をインポート
from function_calculation import refractive, solve_T, calc_temp, plot_temp, cut_2d_array_by_threshold, approximation_cutoff_temp

import streamlit as st
# @st.cache_data
def make_temp_dict(phase_dict, x_axis, d_micro_to_pix_temp, n_room,  Nx, Nz, l, lamda):
    #* temperature extraction
    temp_full_arrey_dict = {}
    r_dict = {}
    for key, value in phase_dict.items():
        res1, res2, x_axis_pix_half = calc_temp(value,x_axis, Nx, Nz, 0, l, d_micro_to_pix_temp, n_room, lamda)
        temp_full_arrey_dict[key] = res1
        r_dict[key] = res2

    temp_full_fig_dict = { key: plot_temp(value, d_micro_to_pix_temp) for key, value in temp_full_arrey_dict.items()}
    return temp_full_arrey_dict, temp_full_fig_dict, r_dict, x_axis_pix_half

#TODO 高さによってデータ数が０の部分があるので、要調整
def make_temp_apr_dict(temp_full_array_cutoff_dict, x_axis_pix_cutoff_dict, n_apr_pix, T_room):
    # 結果を保存する辞書を初期化
    temp_apr_dict = {}
    popt_full_dict = {}
    popt_init_full_dict = {}
    # del temp_full_array_cutoff_dict['offset']  # 'offset_convolve'を削除
    # del temp_full_array_cutoff_dict['original']  # 'offset_convolve'を削除
    # del temp_full_array_cutoff_dict['offset_convolve_shift']  # 'offset_convolve'を削除
    for key in temp_full_array_cutoff_dict:
        if key in x_axis_pix_cutoff_dict:
            temp_cutoff = temp_full_array_cutoff_dict[key]
            x_axis_pix_cutoff = x_axis_pix_cutoff_dict[key]

            temp_apr, popt_full, popt_init_full = approximation_cutoff_temp(
                temp_cutoff, x_axis_pix_cutoff, n_apr_pix, T_room)

            # 各結果を辞書に保存
            temp_apr_dict[key] = temp_apr
            popt_full_dict[key] = popt_full
            popt_init_full_dict[key] = popt_init_full

    # 複数辞書をまとめて返す
    return temp_apr_dict, popt_full_dict, popt_init_full_dict

class TempAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, phase_full_arrey_dict, x_axis,  d_micro_to_pix_temp, k_extract_pix_phase_from_top, convolve_size_temp, Nx, Nz, l, n_apr_pix, lamda, T_room, target, threshold_cutoff):
        self.phase_dict = phase_full_arrey_dict
        self.x_axis = x_axis
        self.d_micro_to_pix_temp = d_micro_to_pix_temp
        self.k_extract_pix_phase_from_top = k_extract_pix_phase_from_top
        self.convolve_size_temp = convolve_size_temp
        self.Nx = Nx
        self.Nz = Nz
        self.l = l
        self.n_apr_pix = n_apr_pix #* 近似に使用するフレーム数
        self.lamda = lamda
        self.T_room = T_room
        self.target = target
        self.threshold_cutoff = threshold_cutoff

        #*水の常温=室温として，室温における水の屈折率を算出
        self.n_room = refractive(self.T_room)

        self.temp_full_arrey_dict, self.temp_full_fig_dict, self.r_dict, self.x_axis_pix_half = make_temp_dict(
            phase_dict = self.phase_dict,
            x_axis = self.x_axis,
            d_micro_to_pix_temp = self.d_micro_to_pix_temp,
            n_room = self.n_room,
            Nx = self.Nx,
            Nz = self.Nz,
            l = self.l,
            lamda = self.lamda
        )

        #* 温度の辞書から'_centered'を含むキーのみ抽出
        self.temp_full_arrey_dict = { key: value for key, value in self.temp_full_arrey_dict.items() if '_centered' in key }

        #* scipyを使い、高さに対して移動平均を取る
        #! size = 1 にしても、元の二次元配列と全く同じものが返るわけではない。
        self.temp_full_arrey_uniform_dict = { key: uniform_filter1d(value, size=10, axis=0, mode='nearest') for key, value in self.temp_full_arrey_dict.items() }

        #* 温度分布cutoff処理
        #? _uniform_する前の温度の辞書
        self.temp_full_array_cutoff_dict = { key: cut_2d_array_by_threshold(value, self.target, self.threshold_cutoff, from_end=False) for key, value in self.temp_full_arrey_dict.items() }

        #? _uniform_した後の温度の辞書
        self.temp_full_array_uniform_cutoff_dict = { key: cut_2d_array_by_threshold(value, self.target, self.threshold_cutoff, from_end=False) for key, value in self.temp_full_arrey_uniform_dict.items() }

        #* cutoff処理の温度分布に合わせてx軸の切り取り、辞書作成
        #? _uniform_する前
        self.x_axis_pix_cutoff_dict = {}
        for key, array_2d in self.temp_full_array_cutoff_dict.items():
            result = []
            for row in array_2d:
                length = len(row)
                # x_axis の先頭から length 個取り出す
                x_axis_pix_cut = self.x_axis_pix_half[:length]
                result.append(x_axis_pix_cut)
            # 二次元配列化（行ごとに長さが異なる場合はdtype=objectになる）
            x_axis_pix_array = np.array(result, dtype=object)
            self.x_axis_pix_cutoff_dict[key] = x_axis_pix_array

        #? _uniform_した後

        #* 近似処理
        # self.temp_full_array_cutoff_apr_dict, self.popt_full_dict, self.popt_init_full_dict = make_temp_apr_dict(self.temp_full_array_cutoff_dict, self.x_axis_pix_cutoff_dict, self.n_apr_pix, self.T_room)
        # temp_apr, popt_full, popt_init_full = approximation_cutoff_temp(
        #     temp_cutoff, x_axis_pix_cutoff, n_apr_pix, T_room)

