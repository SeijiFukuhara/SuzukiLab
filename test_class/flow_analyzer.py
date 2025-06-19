# flow_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import streamlit as st

class FlowAnalyzer:
    def __init__(self, csv_file, d_micro_to_pix=1.0):
        self.d_micro_to_pix = d_micro_to_pix
        self.path = csv_file
        self.df = pd.read_csv(self.path, encoding="cp932", skiprows=7, usecols=range(1, 7), engine='python')
        self.li_T_f = self._prepare_data()
        self._generate_grid()

    #* 読み取ったcsvファイルを、計算に使えるように処理
    def _prepare_data(self):
        df_T = self.df.T
        df_T_replace = df_T.replace("-", 0) #? ファイル中の"-"を0に置き換え
        li_T = df_T_replace.to_numpy().tolist() 
        return np.vectorize(float)(li_T) #? リストや配列の各要素をすべてfloat型に変換

    def _generate_grid(self):
        '''
        li = [
            [10, 20, 30],   # li[0] x座標情報
            [1, 2, 3]       # li[1] y座標情報
        ]
        li_gridpoint = list(zip(li[0], li[1]))
        => [(10, 1), (20, 2), (30, 3)]
        '''
        li = self.li_T_f
        li_gridpoint = list(zip(li[0], li[1])) #? x座標とy座標をベアにしてタプルのリストを作成
        li_coordinates = list(zip(li[2], li[3]))
        li_velocity = list(zip(li[4], li[5]))
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
        self.gridpoint_velocity = {k: v for k, v in zip(li_gridpoint, li_velocity)}
        self.coordinates_gridpoint = {k: v for k, v in zip(li_coordinates, li_gridpoint)}
        #? 座標が(0, 0)のグリッドポイントを取得（バブル中心のグリッドポイントを取得）
        self.x0, self.y0 = map(int, self.coordinates_gridpoint[(0.0, 0.0)])

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
