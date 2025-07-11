import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from scipy.ndimage import uniform_filter1d
import pickle

#? 関数ファイルから必要な関数をインポート
from function_calculation import refractive, solve_T, calc_temp, plot_temp, cut_2d_array_by_threshold, approximation_cutoff_temp, nan_below_threshold_2d

# import streamlit as st
# @st.cache_data
def make_temp_dict(phase_dict, x_axis, d_micro_to_pix_temp, n_room,  Nx, Nz, l, lamda):
    #* temperature extraction
    temp_full_array_dict = {}
    r_dict = {}
    for key, value in phase_dict.items():
        res1, res2, x_axis_pix_half = calc_temp(value,x_axis, Nx, Nz, 0, l, d_micro_to_pix_temp, n_room, lamda)
        temp_full_array_dict[key] = res1
        r_dict[key] = res2

    temp_full_fig_dict = { key: plot_temp(value, d_micro_to_pix_temp) for key, value in temp_full_array_dict.items()}
    return temp_full_array_dict, temp_full_fig_dict, r_dict, x_axis_pix_half

#TODO 高さによってデータ数が０の部分があるので、要調整
def make_temp_apr_dict(temp_full_array_cutoff_dict, x_axis_pix_half, n_apr_pix, T_room, min_points_required):
    # 結果を保存する辞書を初期化
    temp_apr_dict = {}
    popt_full_dict = {}
    popt_init_full_dict = {}
    skipped_indices_full_dict = {}
    for key in temp_full_array_cutoff_dict:
        temp_cutoff = temp_full_array_cutoff_dict[key]

        temp_apr, popt_full, popt_init_full, skipped_indices = approximation_cutoff_temp(
            temp_cutoff, x_axis_pix_half, n_apr_pix, T_room, min_points_required)

        # 各結果を辞書に保存
        temp_apr_dict[key] = temp_apr
        popt_full_dict[key] = popt_full
        popt_init_full_dict[key] = popt_init_full
        skipped_indices_full_dict[key] = skipped_indices


    # 複数辞書をまとめて返す
    return temp_apr_dict, popt_full_dict, popt_init_full_dict, skipped_indices_full_dict

class TempAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, phase_full_array_dict, x_axis,  d_micro_to_pix_temp, k_extract_pix_phase_from_top, convolve_size_temp, Nx, Nz, l, n_apr_pix,h0, lamda, T_room, target, threshold_cutoff, uniform_filter_size, min_points_required):
        self.phase_dict = phase_full_array_dict
        self.x_axis = x_axis
        self.d_micro_to_pix_temp = d_micro_to_pix_temp
        self.k_extract_pix_phase_from_top = k_extract_pix_phase_from_top
        self.convolve_size_temp = convolve_size_temp
        self.Nx = Nx
        self.Nz = Nz
        self.l = l
        self.n_apr_pix = n_apr_pix #* 近似に使用するフレーム数
        self.h0 = h0
        self.lamda = lamda
        self.T_room = T_room
        self.target = target
        self.threshold_cutoff = threshold_cutoff
        self.uniform_filter_size = uniform_filter_size
        self.min_points_required = min_points_required

        #*水の常温=室温として，室温における水の屈折率を算出
        self.n_room = refractive(self.T_room)

        self.temp_full_array_dict, self.temp_full_fig_dict, self.r_dict, self.x_axis_pix_half = make_temp_dict(
            phase_dict = self.phase_dict,
            x_axis = self.x_axis,
            d_micro_to_pix_temp = self.d_micro_to_pix_temp,
            n_room = self.n_room,
            Nx = self.Nx,
            Nz = self.Nz,
            l = self.l,
            lamda = self.lamda
        )


        with open('temp_full_array_dict.pkl', 'wb') as f:
            pickle.dump(self.temp_full_array_dict, f)

        with open('temp_full_fig_dict.pkl', 'wb') as f:
            pickle.dump(self.temp_full_fig_dict, f)

        with open('x_axis_pix_half.pkl', 'wb') as f:
            pickle.dump(self.x_axis_pix_half, f)


        #* 温度の辞書から'_centered'を含むキーのみ抽出
        self.temp_full_array_dict = { key: value for key, value in self.temp_full_array_dict.items() if '_centered' in key }

        #* scipyを使い、高さに対して移動平均を取る(unifrom処理)
        #! size = 1 にしても、元の二次元配列と全く同じものが返るわけではない。
        self.temp_full_array_uniform_dict = { key: uniform_filter1d(value, size=uniform_filter_size, axis=0, mode='nearest') for key, value in self.temp_full_array_dict.items() }

        #* 温度分布cutoff処理
        #! 温度がT_roomと一致する領域と基板領域にNaN埋めを実行。し二次元アレイの大きさは1024×1124を維持。
        #? _uniform_する前の温度のcutoff辞書
        # self.temp_full_array_cutoff_dict = { key: cut_2d_array_by_threshold(value, self.target, self.threshold_cutoff, self.h0, from_end=True) for key, value in self.temp_full_array_dict.items() }
        self.temp_full_array_cutoff_dict = { key: nan_below_threshold_2d(value, self.T_room) for key, value in self.temp_full_array_dict.items() }

        #? _uniform_した後の温度のcutoff辞書
        # self.temp_full_array_uniform_cutoff_dict = { key: cut_2d_array_by_threshold(value.copy(), target, threshold_cutoff, h0, from_end=True) for key, value in self.temp_full_array_uniform_dict.items() }
        self.temp_full_array_uniform_cutoff_dict = { key: nan_below_threshold_2d(value, self.T_room) for key, value in self.temp_full_array_uniform_dict.items() }


        #* 近似処理
        self.temp_full_array_cutoff_apr_dict, self.popt_full_dict, self.popt_init_full_dict, self.skipped_indices_full_dict = make_temp_apr_dict(self.temp_full_array_cutoff_dict,self.x_axis_pix_half, n_apr_pix, T_room, min_points_required)
        self.temp_full_array_uniform_cutoff_apr_dict, self.popt_full_uniform_dict, self.popt_init_full_uniform_dict, self.skipped_indices_full_uniform_dict = make_temp_apr_dict(self.temp_full_array_cutoff_dict,self.x_axis_pix_half, n_apr_pix, T_room, min_points_required)

