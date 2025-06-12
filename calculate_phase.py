from function_calculation import loadtext, offset, plot_phase, plot_phase_and_save
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import matplotlib.pyplot as plt
from function_calculation import extract_frame_range_suffix
import sys
import os
from datetime import datetime

d_temp = 1.9833
convolve_size_temp = 21 #*移動平均サイズ
z1 = 500
z2 = 510
x1 = 0
x2 = 20

input_folder = sys.argv[1]
start_frame, end_frame = extract_frame_range_suffix(input_folder)
# 現在の日時を取得してフォーマット（例: "_20250611_1930"）
timestamp_str = datetime.now().strftime("_%Y%m%d_%H%M")

# 親ディレクトリを取得（末尾除く）
parent_dir = os.path.dirname(input_folder)
output_png_folder = os.path.join(parent_dir, "phase_png_frames_" + str(start_frame) + "_" + str(end_frame) + timestamp_str)
output_bmp_folder = os.path.join(parent_dir, "phase_bmp_frames_" + str(start_frame) + "_" + str(end_frame) + timestamp_str)
os.makedirs(output_png_folder, exist_ok=True)
os.makedirs(output_bmp_folder, exist_ok=True)

for fname in sorted(os.listdir(input_folder)):
    if fname.lower().endswith('.csv'):
        csv_path = os.path.join(input_folder, fname)
        img_phase_array = loadtext(csv_path)
        img_phase_array_offset = offset(img_phase_array, 1,convolve_size_temp,z1,z2,x1,x2)

        # 保存先のパスを設定（入力CSV名と同じ名前で .bmp にする）
        # output_path = fname_phase.replace('.csv', '.bmp')
        plot_phase_and_save(img_phase_array_offset, d_temp,csv_path, output_png_folder, output_bmp_folder)



# img_phase_array = loadtext(fname_phase)
# img_phase_array_offset = offset(img_phase_array, 1,convolve_size_temp,z1,z2,x1,x2)

# # 保存先のパスを設定（入力CSV名と同じ名前で .bmp にする）
# # output_path = fname_phase.replace('.csv', '.bmp')
# plot_phase_and_save(img_phase_array_offset, d_temp, fname_phase)

print(f"✅ 画像が保存されました:")