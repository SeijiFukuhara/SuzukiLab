import numpy as np
import sys
from configparser import ConfigParser
import time
import PIL.Image
import os
import re
from datetime import datetime
import time

from extract_phase import get_phase,get_phase_from_reference, unwrap_phase, save_array, save_video, _video2images
from function_calculation import extract_frame_range_suffix, add_tilde_to_filename, find_available_filename, video2images_rewrite, _video2images, load_video_with_leading_image
# from function_calculation import loadtext, offset, plot_phase, plot_phase_and_save
try:
    import cv2
    __HAS_OPENCV__ = True
except ImportError:
    __HAS_OPENCV__ = False

__VIDEO_EXTENSIONS__ = ['.avi', '.mp4']

if __name__ == '__main__':
    if len(sys.argv) < 2: #* コマンドプロンプトからの実行時に引数がない場合は、エラーを出力
        raise ValueError(
            'Usage: get_phase.py [path/to/image] '
            'or \nget_phase.py [path/to/reference_image] [path/to/target_image1] ... '
            '\n  [NOTE] target_image1 can include the wild cards.'
            '\n         e.g. step[0-9][0-9].bmp will cover step01.bmp, step02.bmp, ..., step99.bmp'
            'or \nget_phase.py [path/to/video] reference_frames(e.g. 1,2 or 1-10)'
        )

    #* "20250614_183045" のような日付＋時刻
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = ConfigParser()
    config.read('settings.ini')
    image_format = config['output']['format'].strip()
    video_format = config['output'].get('video_format').strip()
    if ',' in image_format:
        image_format = [img_format.strip() for img_format in image_format.split(',')]

    path_refimage = sys.argv[1] #* フレーム数：1枚
    path_video = sys.argv[2] #* フレーム数：n枚
    video_images = None
    started = time.time()

    # ディレクトリのチェック
    dir_ref = os.path.abspath(os.path.dirname(path_refimage))
    dir_video = os.path.abspath(os.path.dirname(path_video))
    if dir_ref == dir_video:
        prefix = find_available_filename(path_video)
    else:
        print(" エラー: リファレンス画像 と 変換動画 は同じディレクトリに存在しません。")
        print(f"  リファレンス画像 のディレクトリ: {dir_ref}")
        print(f"  変換動画 のディレクトリ: {dir_video}")


    #* 入力ビデオをバラバラにするlen(video_images) = n
    video_images = video2images_rewrite(path_video)
    image = np.array(PIL.Image.open(path_refimage))

    print(f"変換動画のフレーム数: {len(video_images)}")
    print("変換を開始します...")

    start_time_convert = time.time() # 変換の計測開始

    parameters, convolved = get_phase(np.array(image), config, parameters=None)
    amplitude, phase = np.abs(convolved), np.angle(convolved)

    will_unwrap = config['postprocess']['unwrap'] == 'True'

    image_format = config['output']['video_format'].strip()
    target_amplitudes = []
    target_phases = []

    num_batch = int(config['computation']['num_batch'])
    if num_batch < 0:  # compute full batch
        _, target_conv = get_phase(np.array(video_images), config, parameters=parameters)
    else:
        target_conv = []
        for i in range(0, len(video_images), num_batch):
            target_conv.append(get_phase(
                np.stack(video_images[i: i + num_batch], axis=0), 
                config, parameters=parameters)[1]
            )
        target_conv = np.concatenate(target_conv, axis=0)

    target_phases = get_phase_from_reference(target_conv, convolved)
    if will_unwrap:
        target_phases = np.clip(
            unwrap_phase(target_phases),
            float(config['postprocess']['unwrap_phase_min']),
            float(config['postprocess']['unwrap_phase_max'])
        )
    #* ファイル名の先頭に prefix を追加（なくてもよい）
    path_video = add_tilde_to_filename(path_video, prefix)
    save_video(target_phases, path_video, '_phase', video_format='avi')


    end_time_convert = time.time()  # 終了時刻
    elapsed_time_convert = end_time_convert - start_time_convert

    print("変換が完了しました。")
    print(f"変換動画保存先: {path_video}")
    print(f"出力動画フレーム数：{len(target_phases)}")
    print(f"処理時間: {elapsed_time_convert:.2f}秒")

    # print(target_phases.shape)


    print("CSVファイルを保存します...")
    start_time_csv = time.time()  # CSV保存の計測開始

    # 開始・終了フレーム番号を抽出
    start_frame, end_frame = extract_frame_range_suffix(path_video)
    # 保存するディレクトリを作成
    video_dir = os.path.dirname(path_video)
    output_dir = os.path.join(video_dir, "phase_csv_frames_" + str(start_frame) + "_" + str(end_frame))
    output_dir = add_tilde_to_filename(output_dir, prefix)  #* ファイル名の先頭に prefix を追加（なくてもよい）
    os.makedirs(output_dir, exist_ok=True)
    
    # ベースとなる src_path を定義（ダミー拡張子でOK）
    base_path = os.path.join(output_dir, "phase.csv")

    # 各スライスを保存
    for i in range(target_phases.shape[0]):
        suffix = f"_{start_frame + i:04d}"
        save_array(target_phases[i], base_path, suffix, image_format='csv')

    end_time_csv = time.time()  # 終了時刻
    elapsed_time_csv = end_time_csv - start_time_csv
    print(f"{target_phases.shape[0]} 枚のCSVファイルを {output_dir}/ に保存しました。")
    print(f"CSV保存時間: {elapsed_time_csv:.2f}秒")

    with open("settings.ini", "r", encoding="utf-8") as f:
        settings_text = f.read()

    log_path = os.path.join(video_dir, "conversion.log")    
    log_path = add_tilde_to_filename(log_path, prefix)  #* ファイル名の先頭に prefix を追加（なくてもよい）

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=== 変換時刻 ===\n")
        f.write(timestamp + "\n")
        f.write("=== 入力 ===\n")
        f.write(f"使用プログラム：{sys.argv[0]}\n")
        f.write(f"リファレンス：{sys.argv[1]}\n")
        f.write(f"変換動画：{sys.argv[2]}\n")
        f.write("=== 出力 ===\n")
        f.write(f"変換動画保存先: {path_video}秒\n")
        f.write(f"変換処理時間: {elapsed_time_convert:.2f}秒\n")
        f.write(f"csv保存先: {output_dir}秒\n")
        f.write(f"csv処理時間: {elapsed_time_csv:.2f}秒\n")
        f.write("=== 使用した設定ファイル内容 ===\n")
        f.write(settings_text)
        f.write("\n=============================\n")

    print(f"ログファイルを {log_path} に保存しました。")