import numpy as np


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

#? 関数ファイルから必要な関数をインポート
from function_calculation import cut_2d_array_by_threshold, approximation_cutoff

def malke_temp_apr_dict(temp_full_array_cutoff_dict, x_axis_pix_cutoff_dict, n_apr_pix, T_room):
    # 結果を保存する辞書を初期化
    temp_apr_dict = {}
    popt_full_dict = {}
    popt_init_full_dict = {}
    del temp_full_array_cutoff_dict['offset']  # 'offset_convolve'を削除
    # del temp_full_array_cutoff_dict['offset_convolve_shift']  # 'offset_convolve'を削除
    for key in temp_full_array_cutoff_dict:
        if key in x_axis_pix_cutoff_dict:
            temp_cutoff = temp_full_array_cutoff_dict[key]
            x_axis_pix_cutoff = x_axis_pix_cutoff_dict[key]

            temp_apr, popt_full, popt_init_full = approximation_cutoff(
                temp_cutoff, x_axis_pix_cutoff, n_apr_pix, T_room, mode='temp'
            )

            # 各結果を辞書に保存
            temp_apr_dict[key] = temp_apr
            popt_full_dict[key] = popt_full
            popt_init_full_dict[key] = popt_init_full

    # 複数辞書をまとめて返す
    return temp_apr_dict, popt_full_dict, popt_init_full_dict
class HeatFluxAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, dict_temp, x_axis_pix_half,  k_extract_pix_from_top, target, threshold, n_apr_pix, T_room):
        self.dict_temp = dict_temp
        self.x_axis_pix_half = x_axis_pix_half  # x軸の半分の配列
        # self.dict_flow = dict_flow.
        self.k_extract_pix_from_top = k_extract_pix_from_top
        self.target = target
        self.threshold = threshold
        self.n_apr_pix = n_apr_pix  # 近似に使用するフレーム数
        self.T_room = T_room


        #*cutoff処理
        self.temp_full_array_cutoff_dict = { key: cut_2d_array_by_threshold(value, self.target, self.threshold, from_end=False) for key, value in self.dict_temp.items() }

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

        #* 近似処理
        self.temp_full_array_cutoff_apr_dict, self.popt_full_dict, self.popt_init_full_dict = malke_temp_apr_dict(self.temp_full_array_cutoff_dict, self.x_axis_pix_cutoff_dict, self.n_apr_pix, self.T_room)