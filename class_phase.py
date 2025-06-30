import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

#? 関数ファイルから必要な関数をインポート
from function_calculation import offset, plot_phase,approximation_phase

class PhaseAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, csv_file, d_micro_to_pix_temp, k_extract_microm_phase, convolve_size_temp,Nx, Nz, n_apr_pix, z1, z2, x1, x2, gaussian_additive_term):
        self.path = csv_file
        self.d_micro_to_pix_temp = d_micro_to_pix_temp
        self.k_extract_microm_phase = k_extract_microm_phase
        self.convolve_size_temp = convolve_size_temp
        self.Nx = Nx
        self.Nz = Nz
        self.n_apr_pix = n_apr_pix #* 近似に使用するフレーム数
        self.z1 = z1
        self.z2 = z2
        self.x1 = x1
        self.x2 = x2
        self.gaussian_additive_term = gaussian_additive_term

        #* csvファイル読み込み
        self.df = pd.read_csv(self.path, encoding="cp932",index_col = None, header = None, engine = 'python')
        img_phase_array = self.df.to_numpy()

        self.width_phase = len(img_phase_array[0]) #*画像の横幅[pix](1024)
        self.height_phase = len(img_phase_array) #*画像の縦幅[pix](1024)
        self.x_axis = np.arange(-Nx, -Nx + self.width_phase)

        #* offset
        self.img_phase_array_offset = offset(img_phase_array, self.convolve_size_temp, self.z1, self.z2, self.x1, self.x2, convolve = False)
        self.fig_offset = plot_phase(self.img_phase_array_offset, self.d_micro_to_pix_temp)
        #* offset and convolve
        self.img_phase_array_offset_convolve = offset(img_phase_array, self.convolve_size_temp, self.z1, self.z2, self.x1, self.x2, convolve = True)
        self.fig_offset_convolve = plot_phase(self.img_phase_array_offset_convolve, self.d_micro_to_pix_temp)

        try:
            self.phase_full, self.popt_full, self.popt_init_full = approximation_phase(
                twolist_array = self.img_phase_array_offset_convolve,
                x_axis = self.x_axis,
                width_phase = self.width_phase,
                height_phase = self.height_phase,
                n_apr_pix = self.n_apr_pix,
                gaussian_additive_term = self.gaussian_additive_term,
                )
            self.fig_offset_convolve_gaussian_plus_linear = plot_phase(self.phase_full[0], self.d_micro_to_pix_temp)
            self.fig_offset_convolve_gaussian = plot_phase(self.phase_full[1], self.d_micro_to_pix_temp)
            self.fig_offset_convolve_gaussian_centered = plot_phase(self.phase_full[2], self.d_micro_to_pix_temp)
        except Exception as e:
            # 例外が発生したときの処理
            self.error_msg = traceback.format_exc()
