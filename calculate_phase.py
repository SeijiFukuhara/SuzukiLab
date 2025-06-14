from function_calculation import loadtext, offset, plot_phase, plot_phase_and_save
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import matplotlib.pyplot as plt
from function_calculation import extract_frame_range_suffix
import sys
import os
import time

from datetime import datetime
from function_calculation import add_tilde_to_filename, find_available_filename_combination

d_temp = 1.9833
convolve = False #*移動平均をとるかどうか
convolve_size_temp = 21 #*移動平均サイズ
z1 = 500
z2 = 510
x1 = 0
x2 = 20

csv_folder = sys.argv[1]
start_frame, end_frame = extract_frame_range_suffix(csv_folder)
prefix = find_available_filename_combination(csv_folder)

# 現在の日時を取得してフォーマット（例: "_20250611_1930"）
timestamp = datetime.now().strftime("_%Y%m%d_%H%M")

# 親ディレクトリを取得（末尾除く）
parent_dir = os.path.dirname(csv_folder)
# output_png_folder = os.path.join(parent_dir, "phase_png_frames_" + str(start_frame) + "_" + str(end_frame) + timestamp_str)
output_bmp_folder = os.path.join(parent_dir, "phase_bmp_frames_" + str(start_frame) + "_" + str(end_frame))
output_bmp_folder = add_tilde_to_filename(output_bmp_folder, prefix)  #* ファイル名の先頭に prefix を追加（なくてもよい）
# os.makedirs(output_png_folder, exist_ok=True)
os.makedirs(output_bmp_folder, exist_ok=True)

start_time_phase = time.time() # 変換の計測開始


for fname in sorted(os.listdir(csv_folder)): #? 指定されたフォルダ内の.csvファイルを一つずつ読み取り
    if fname.lower().endswith('.csv'):
        csv_path = os.path.join(csv_folder, fname)
        img_phase_array = loadtext(csv_path)
        img_phase_array_offset = offset(img_phase_array, convolve,convolve_size_temp,z1,z2,x1,x2)

        # 保存先のパスを設定（入力CSV名と同じ名前で .bmp にする）
        # output_path = fname_phase.replace('.csv', '.bmp')
        plot_phase_and_save(img_phase_array_offset, d_temp,csv_path, output_bmp_folder)


end_time_phase = time.time()  # 終了時刻
elapsed_time_phase = end_time_phase - start_time_phase

print(f"画像が保存されました:")
print(f"処理時間: {elapsed_time_phase:.2f}秒")


csv_dir = os.path.dirname(csv_folder)


log_path = os.path.join(csv_dir, "phase.log")
log_path = add_tilde_to_filename(log_path, prefix)  #* ファイル名の先頭に prefix を追加（なくてもよい）

convolve = True #*移動平均をとるかどうか
convolve_size_temp = 21 #*移動平均サイズ
z1 = 500
z2 = 510
x1 = 0
x2 = 20




with open(log_path, "a", encoding="utf-8") as f:
    f.write("=== 変換時刻 ===\n")
    f.write(timestamp + "\n")
    f.write("=== 入力 ===\n")
    f.write(f"使用プログラム：{sys.argv[0]}\n")
    f.write(f"csvフォルダ名：{sys.argv[1]}\n")
    f.write("=== 出力 ===\n")
    f.write(f"位相画像保存先: {output_bmp_folder}秒\n")
    f.write(f"変換処理時間: {elapsed_time_phase:.2f}秒\n")
    f.write("=== 使用した設定内容 ===\n")
    f.write(f"変換比率 d_temp：{d_temp}\n")
    f.write(f"移動平均を取るかどうか convolve：{convolve}\n")
    f.write(f"移動平均サイズ convolve_size_temp：{convolve_size_temp}\n")
    f.write(f"室温領域 [z1:z2,x1:x2]：[{z1}:{z2},{x1}:{x2}]\n")
    f.write("\n=============================\n")