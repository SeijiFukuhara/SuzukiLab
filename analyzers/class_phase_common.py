from function_calculation import make_k_dict
import pandas as pd

class PhaseCommon:
    def __init__(self, fname_phase, k_extract_microm, debug_k_pix, radio_microm_or_pix, d_microm_to_pix_temp, height_phase, h0):

        temp_phase_path = "temp_uploaded.csv"
        with open(temp_phase_path, "wb") as f:
            f.write(fname_phase.read())

        #* csvファイル読み込み
        df = pd.read_csv(temp_phase_path, encoding="cp932",index_col = None, header = None, engine = 'python')
        img_phase_array = df.to_numpy()

        #* 画像の横幅、縦幅の取得、x軸の作成
        height_phase = img_phase_array.shape[0]
        width_phase = img_phase_array.shape[0]

        self.k_extract_phase_dict = make_k_dict(k_extract_microm, debug_k_pix, radio_microm_or_pix, d_microm_to_pix_temp, height_phase, h0)