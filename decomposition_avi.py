import cv2
import os
import sys

#! 指定された .avi 動画ファイルから各フレームを抽出し、グレースケールの .bmp 画像として保存するツールです。

def extract_frames_as_bmp(video_path):
    # 入力ファイルのディレクトリとファイル名（拡張子なし）を取得
    input_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # 出力フォルダ名（同じディレクトリ内に "base_name_frames" フォルダを作る）
    output_folder = os.path.join(input_dir, f"{base_name}_frames")
    os.makedirs(output_folder, exist_ok=True)

    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    success, frame = cap.read()

    while success:
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 保存先のパス
        output_path = os.path.join(output_folder, f"{base_name}_{frame_index:04d}.bmp")
        cv2.imwrite(output_path, gray)

        success, frame = cap.read()
        frame_index += 1

    cap.release()
    print(f"{frame_index} フレームを保存しました: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python extract_frames_cli.py input_video.avi")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"エラー: ファイルが存在しません: {video_path}")
        sys.exit(1)

    extract_frames_as_bmp(video_path)
    print("終わったよ！！！！！１")