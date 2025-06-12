import cv2
import numpy as np
import os
import sys

def avi_to_csv_frames(video_path):
    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画を開けませんでした -> {video_path}")
        return

    # 出力ディレクトリの作成
    video_dir = os.path.dirname(video_path)
    output_dir = os.path.join(video_dir, "csv_frames")
    os.makedirs(output_dir, exist_ok=True)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # フレームを CSV に保存
        csv_filename = f"frame_{frame_index:04d}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        np.savetxt(csv_path, gray, fmt='%d', delimiter=',')  # 整数で保存

        frame_index += 1

    cap.release()
    print(f"{frame_index} 枚のフレームをCSVとして保存しました（保存先: {output_dir}）")

# --- 実行部分 ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使い方: python avi_to_csv_frames.py 動画ファイル.avi")
    else:
        avi_to_csv_frames(sys.argv[1])
