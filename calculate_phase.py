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
offset_area_slice = (slice(z1, z2), slice(x1, x2)) #* [z1:z2,x1:x2]の範囲の位相を平均し，その位相を0にoffset，絶対水温の領域を指定．zは縦方向，xは横方向．順番に注意．

csv_folder = sys.argv[1]
start_frame, end_frame = extract_frame_range_suffix(csv_folder)
prefix = find_available_filename_combination(csv_folder)

#* 大文字と小文字を区別しないようにする
convolve = sys.argv[2].lower() == 'true'  # コマンドライン引数から移動平均の有無を取得

# 現在の日時を取得してフォーマット（例: "_20250611_1930"）
timestamp = datetime.now().strftime("_%Y%m%d_%H%M")

# 親ディレクトリを取得（末尾除く）
parent_dir = os.path.dirname(csv_folder)
# output_png_folder = os.path.join(parent_dir, "phase_png_frames_" + str(start_frame) + "_" + str(end_frame) + timestamp_str)
output_bmp_folder = os.path.join(parent_dir, "phase_bmp_frames_" + str(start_frame) + "_" + str(end_frame))
output_bmp_folder = add_tilde_to_filename(output_bmp_folder, prefix)  #* ファイル名の先頭に prefix を追加（なくてもよい）
# os.makedirs(output_png_folder, exist_ok=True)
os.makedirs(output_bmp_folder, exist_ok=True)

#* 時間計測開始
start_time_phase = time.time()


# .csvファイル一覧取得
csv_files = sorted([f for f in os.listdir(csv_folder) if f.lower().endswith('.csv')])
total_files = len(csv_files)

# 各ファイルに対して処理
for i, fname in enumerate(csv_files, start=1):
    csv_path = os.path.join(csv_folder, fname)
    img_phase_array = loadtext(csv_path)

    # offset処理
    img_phase_array_offset = offset(img_phase_array, convolve_size_temp, offset_area_slice, convolve=convolve)

    # BMP画像として保存
    plot_phase_and_save(img_phase_array_offset, d_temp, csv_path, output_bmp_folder)

    # 上書きで進捗表示
    percent = (i / total_files) * 100
    print(f"\r{i}/{total_files} ({percent:.1f}%) 完了: {fname}", end="", flush=True)

#* 改行を明示（最後に表示が次の行へ移動）
print()

#* 時間計測終了
end_time_phase = time.time()  # 終了時刻
elapsed_time_phase = end_time_phase - start_time_phase

print(f"画像が保存されました:")
print(f"処理時間: {elapsed_time_phase:.2f}秒")


csv_dir = os.path.dirname(csv_folder)


log_path = os.path.join(csv_dir, "phase.log")
log_path = add_tilde_to_filename(log_path, prefix)  #* ファイル名の先頭に prefix を追加（なくてもよい）

with open(log_path, "a", encoding="utf-8") as f:
    f.write("=== 変換時刻 ===\n")
    f.write(timestamp + "\n")
    f.write("=== 入力 ===\n")
    f.write(f"使用プログラム：{sys.argv[0]}\n")
    f.write(f"csvフォルダ名：{sys.argv[1]}\n")
    f.write("=== 出力 ===\n")
    f.write(f"位相画像保存先: {output_bmp_folder}\n")
    f.write(f"変換処理時間: {elapsed_time_phase:.2f}秒\n")
    f.write("=== 使用した設定内容 ===\n")
    f.write(f"変換比率 d_temp：{d_temp}\n")
    f.write(f"移動平均を取るかどうか convolve：{convolve}\n")
    f.write(f"移動平均サイズ convolve_size_temp：{convolve_size_temp}\n")
    f.write(f"室温領域 [z1:z2,x1:x2]：[{z1}:{z2},{x1}:{x2}]\n")
    f.write("\n=============================\n")