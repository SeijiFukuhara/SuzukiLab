#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# flow_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

#! 格子間隔は1[pix]の前提。そうでない場合は改変の必要あり。
class FlowAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, csv_file, d_micro_to_pix_flow, k_extract_microm_flow, adjust_x_grid, convolve_size_flow, debug_k_pix, microm_or_pix):

        self.path = csv_file
        self.d_micro_to_pix_flow = d_micro_to_pix_flow #* 流速分布観察カメラの1umあたりのpixel d[pixel/μm]
        self.k_extract_microm_flow = k_extract_microm_flow #* 流速分布観察で抽出するy座標の高さ [μm]
        self.adjust_x_grid = adjust_x_grid
        self.convolve_size_flow = convolve_size_flow

        #* csvファイル読み込み
        self.df = pd.read_csv(self.path, encoding="cp932", skiprows = 7, skipfooter = 0, usecols = range(1,7), index_col = None, header = None, engine = 'python')

        #* 高さ、幅を取得
        self.flow_ar, self.width_grid, self.height_grid = self._prepare_data()
        #? バブル中心の座標をFE座標系で取得
        self.x0_grid, self.y0_grid = self._generate_grid()
        #? FE座標系でのx座標配列（バブル中心を原点とし、微調整を加える）
        self.x0_new = self.x0_grid + self.adjust_x_grid  # 原点の位置を微調整
        self.x_axis_grid = np.arange(self.width_grid) - self.x0_new  # x座標配列を生成

        #* microm_or_pixの選択
        #? pixを直接指定するか、μmを指定するか選択
        if microm_or_pix == "k_extract_microm[μm]":
            self.k_extract_pix_flow_from_substrate =  int(self.k_extract_microm * self.d_micro_to_pix_flow)
            self.k_extract_pix_flow_from_bottom = self.k_extract_pix_flow_from_substrate + self.y0_grid
            self.k_extract_microm_flow_from_substrate = self.k_extract_microm
            self.k_extract_microm_flow_from_bottom =round(self.k_extract_pix_flow_from_bottom / self.d_micro_to_pix_flow, 2)
        elif microm_or_pix == "debug_k_pix[pix]":
            self.k_extract_pix_flow_from_bottom = debug_k_pix
            self.k_extract_pix_flow_from_substrate = self.k_extract_pix_flow_from_bottom - self.h0
            self.k_extract_microm_flow_from_bottom =  round(self.k_extract_pix_flow_from_bottom / self.d_micro_to_pix_flow,2)
            self.k_extract_microm_flow_from_substrate = round(self.k_extract_pix_flow_from_substrate / self.d_micro_to_pix_flow,2)
        self.k_extract_pix_flow_from_top = self.height_gird - self.k_extract_pix_flow_from_bottom




        #? x方向とy方向の流速を、FE座標で取得
        self.flow_vx = self.flow_xy(4)
        self.flow_vy = self.flow_xy(5)
        #? y方向の流速の辞書を取得
        self.flow_vy_nest_dict = self.flow_nest_dict(self.flow_vy)

    #* 読み取ったcsvファイルを、計算に使えるように処理
    def _prepare_data(self):
        df_T = self.df.T
        #? ファイル中の"-"を0に置き換え、各要素をすべてfloat型に変換
        df_T_replace = df_T.replace("-", 0.0).astype(float)
        #? numpy配列に変換
        flow_ar = df_T_replace.to_numpy()
        width_grid = int(max(flow_ar[0]))  #? FE格子点の横幅を取得
        height_grid = int(max(flow_ar[1]))  #? FE格子点の高さを取得
        return flow_ar, width_grid, height_grid

    def _generate_grid(self):
        '''
        li = [
            [10, 20, 30],   # li[0] x座標情報
            [1, 2, 3]       # li[1] y座標情報
        ]
        li_gridpoint = list(zip(li[0], li[1]))
        => [(10, 1), (20, 2), (30, 3)]
        '''
        li_gridpoint = list(zip(self.flow_ar[0], self.flow_ar[1])) #? x座標とy座標をベアにしてタプルのリストを作成
        li_coordinates = list(zip(self.flow_ar[2], self.flow_ar[3]))
        li_velocity = list(zip(self.flow_ar[4], self.flow_ar[5]))
        '''
        li_gridpoint   = [(100, 50), (200, 50), (300, 50)] # 実際のグリッド上の座標（物理座標など）
        li_velocity    = [(1.2, 0.1), (1.1, 0.2), (1.0, 0.3)]  # 各点での速度ベクトル
        self.gridpoint_velocity = {
            (100, 50): (1.2, 0.1),
            (200, 50): (1.1, 0.2),
            (300, 50): (1.0, 0.3)
        }
        '''
        #? グリッドポイントと速度ベクトルを結びつける辞書
        gridpoint_velocity = {k: v for k, v in zip(li_gridpoint, li_velocity)}
        coordinates_gridpoint = {k: v for k, v in zip(li_coordinates, li_gridpoint)}
        #? 座標が(0, 0)のグリッドポイントを取得（バブル中心のグリッドポイントを取得）
        x0, y0 = map(int, coordinates_gridpoint[(0.0, 0.0)])
        return x0, y0

    #* 流速を取得（FE座標系）
    def flow_xy(self, col):
        arr_1d = self.flow_ar[col]  # 1次元配列
        arr_2d = np.reshape(arr_1d, (self.height_grid, self.width_grid))  # 2次元化
        return arr_2d

    #* 流速に移動平均をかけるときに使用
    def valid_convolve(self, xx, size):
        b = np.ones(size)/size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(size/2)
        # 補正部分
        xx_mean[0] *= size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= size/(i+n_conv)
            xx_mean[-i] *= size/(i + n_conv - (size % 2))
        # size%2は奇数偶数での違いに対応するため
        return xx_mean

    def rep_convolve(self, li2, func):
        return np.array([func(row, self.convolve_size_flow) for row in li2])

    #* 流速にフィッティングを行う際に使用
    def fit1(self, v_1d):
        def fukuhara_fit(x, a, b, c, d, e):
            return a * np.exp(- (x - b)**2 / (2* c**2)) + d * x + e

        popt, _ = curve_fit(
            fukuhara_fit, self.x_axis_grid, v_1d,
            p0=[0.03, 0.01, 40, 0.001, 0.0001],
            maxfev=20000
        )

        fitted_y = fukuhara_fit(self.x_axis_grid, *popt)
        fitted_y_nobg = popt[0] * np.exp(- (self.x_axis_grid - popt[1])**2 / (2 * popt[2]**2)) + popt[4]

        return fitted_y, fitted_y_nobg, popt

    def fit_repeat(self, v_2d):
        results = [self.fit1(v_2d[i]) for i in range(self.height_grid)]
        ar_flow, ar_flow_nobg, ar_popt = tuple(np.stack(parts) for parts in zip(*results))
        return ar_flow, ar_flow_nobg

    def flow_nest_dict(self, v_2d):
        #? 流速をFE座標で取得
        self.flow_v = v_2d
        #? 平滑化した流速を、FE座標で取得
        self.flow_v_convolve = self.rep_convolve(v_2d, self.valid_convolve)
        #? フィッティングした流速を、FE座標で取得
        self.flow_v_fit, self.flow_v_fit_nobug = self.fit_repeat(v_2d)
        #? 平滑化をフィッティングした流速を、FE座標で取得
        self.flow_v_convolve_fit, self.flow_v_convolve_fit_nobug = self.fit_repeat(v_2d)
        flow_dict = {
            'x': self.x_axis_grid,
            'flow_v': self.flow_v,
            'flow_v_convolve': self.flow_v_convolve,
            'flow_v_fit': self.flow_v_fit,
            'flow_v_convolve_fit': self.flow_v_convolve_fit,
            'flow_v_fit_nobug': self.flow_v_fit_nobug,
            'flow_v_convolve_fit_nobug': self.flow_v_convolve_fit_nobug
        }

        #? 指定された高さの流速を取得
        flow_k_dict = {
            'x': self.x_axis_grid,
            'flow_v_k': self.flow_v[self.k_extract_pix_flow-1],
            'flow_v_convolve_k': self.flow_v_convolve[self.k_extract_pix_flow-1],
            'flow_v_fit_k': self.flow_v_fit[self.k_extract_pix_flow-1],
            'flow_v_convolve_fit_k': self.flow_v_convolve_fit[self.k_extract_pix_flow-1],
            'flow_v_fit_nobug_k': self.flow_v_fit_nobug[self.k_extract_pix_flow-1],
            'flow_v_convolve_fit_nobug_k': self.flow_v_convolve_fit_nobug[self.k_extract_pix_flow-1]
        }

        #? 指定された高さの片側の流速を取得
        flow_k_divided_dict = {
            'x': self.x_axis_grid[self.x0_new:],
            'flow_v_k_divided': self.flow_v[self.k_extract_pix_flow-1][self.x0_new:],
            'flow_v_convolve_k_divided': self.flow_v_convolve[self.k_extract_pix_flow-1][self.x0_new:],
            'flow_v_fit_k_divided': self.flow_v_fit[self.k_extract_pix_flow-1][self.x0_new:],
            'flow_v_convolve_fit_k_divided': self.flow_v_convolve_fit[self.k_extract_pix_flow-1][self.x0_new:],
            'flow_v_fit_nobug_k_divided': self.flow_v_fit_nobug[self.k_extract_pix_flow-1][self.x0_new:],
            'flow_v_convolve_fit_nobug_k_divided': self.flow_v_convolve_fit_nobug[self.k_extract_pix_flow-1][self.x0_new:]
        }

        #? ネストされた辞書を作成
        flow_dict_nest = {
            'flow_dict': flow_dict,
            'flow_k_dict': flow_k_dict,
            'flow_k_divided_dict': flow_k_divided_dict,
        }

        return flow_dict_nest



    '''
    #! 球面上の流速を計算するメソッド群
    def compute_theta_vr(self, r_min, r_max):
        di = {}
        number = 0
        for key, value in self.gridpoint_velocity.items():
            x = key[0] - self.x0
            y = key[1] - self.y0
            r = math.sqrt(x**2 + y**2)
            if y > 0 and r_min <= r <= r_max:
                theta = math.degrees(math.atan2(x, y))
                vr = value[0]*math.sin(math.radians(theta)) + value[1]*math.cos(math.radians(theta))
                di[theta] = vr
                number += 1
        self.theta_vr = di
        return di, number

    def plot_flow(self):
        dic = self.theta_vr
        x, y = zip(*sorted(dic.items()))
        fig = plt.figure()
        plt.plot(x, y, label='測定データ', marker='o', lw=0)
        plt.xlabel("バブル中心軸からの角度[°]", fontname="MS Gothic")
        plt.ylabel("流速[μm/μs]", fontname="MS Gothic")
        plt.legend()
        st.pyplot(fig)

    def plot_approximation(self):
        def sincos_fit(x, a, b, c, d):
            return -a * np.sin(np.radians(b * x + c)) * np.cos(2 * np.radians(b * x + c)) + d

        dic = self.theta_vr
        array_x = np.array(list(dic.keys()))
        array_y = np.array(list(dic.values()))

        popt, _ = curve_fit(sincos_fit, array_x, array_y, p0=[0.03, 1, 90, 0])

        fitted_y = [sincos_fit(x, *popt) for x in array_x]
        fig = plt.figure()
        plt.plot(array_x, array_y, label='測定データ', marker='o', lw=0)
        plt.plot(array_x, fitted_y, label='近似曲線', lw=3)
        plt.xlabel("バブル中心軸からの角度[°]", fontname="MS Gothic")
        plt.ylabel("流速[μm/μs]", fontname="MS Gothic")
        plt.legend()
        st.pyplot(fig)

    def generate_report(self, r_min, r_max, number):
        r_min_um = r_min / self.d_micro_to_pix
        r_max_um = r_max / self.d_micro_to_pix
        report = (
            f"バブル原点のx座標: {self.x0} [pix]\n"
            f"バブル原点のy座標: {self.y0} [pix]\n"
            f"プロットするデータの個数: {number}\n"
            f"取得する半径の範囲: {r_min} <= r <= {r_max} [pix]\n"
            f"取得する半径の範囲: {r_min_um:.2f} <= r <= {r_max_um:.2f} [μm]"
        )
        return report
    '''