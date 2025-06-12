# 動画もしくは画像を上下反転して保存するスクリプト

import cv2
import sys
import os

def flip_and_save_image(image_path):
    """グレースケール画像を上下反転して保存"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"エラー: ファイルが読み込めませんでした -> {image_path}")
        return

    flipped = cv2.flip(img, 0)

    dir_name, base_name = os.path.split(image_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"flip_vertically_{name}{ext}"
    save_path = os.path.join(dir_name, new_name)

    success = cv2.imwrite(save_path, flipped)
    if success:
        print(f"保存完了: {save_path}")
    else:
        print(f"エラー: 保存に失敗しました -> {save_path}")

def flip_and_save_avi(video_path):
    """AVI動画の各フレームを上下反転して保存"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けませんでした -> {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'IYUV')  # IYUV形式（出力用）

    dir_name, base_name = os.path.split(video_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}_flip_vertically.avi"
    save_path = os.path.join(dir_name, new_name)

    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped = cv2.flip(frame, 0)
        out.write(flipped)
        frame_count += 1

    cap.release()
    out.release()
    print(f"{frame_count} フレームを処理しました。保存完了: {save_path}")

def main():
    args = sys.argv[1:]

    if len(args) == 0:
        print("エラー: 入力ファイルを指定してください。")
        return

    # ファイル拡張子の取得（小文字に正規化）
    extensions = [os.path.splitext(path)[1].lower() for path in args]

    if len(args) in [1, 2] and all(ext in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'] for ext in extensions):
        for path in args:
            flip_and_save_image(path)
    elif len(args) == 1 and extensions[0] == '.avi':
        flip_and_save_avi(args[0])
    else:
        print("エラー: 対応しているのは以下の場合です：")
        print("- 静止画（bmp, jpg, png など）を1〜2枚")
        print("- AVIファイルを1つ")

if __name__ == "__main__":
    main()
