import numpy as np


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

#? 関数ファイルから必要な関数をインポート
from function_calculation import cut_array_by_threshold

class HeatFluxAnalyzer:
    #!__init__() 内で self.なしで変数を定義することは可能だが、原則として推奨されない。
    def __init__(self, dict_temp, k_extract_pix_from_top, target, threshold):
        self.dict_phase = dict_temp
        # self.dict_flow = dict_flow.
        self.k_extract_pix_from_top = k_extract_pix_from_top
        self.target = target
        self.threshold = threshold

        self.temp_full_array_cutoff_dict = { key: cut_array_by_threshold(value[k_extract_pix_from_top], target, threshold, from_end=False) for key, value in dict_temp.items() }