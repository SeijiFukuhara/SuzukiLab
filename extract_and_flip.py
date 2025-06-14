import cv2
import os
import argparse

def extract_frame_range(video_path, start_frame, end_frame, flip=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けませんでした -> {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'IYUV')

    if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
        print(f"エラー: 無効なフレーム範囲（動画は {total_frames} フレーム）")
        return

    dir_name, base_name = os.path.split(video_path)
    name, ext = os.path.splitext(base_name)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()

    if not ret:
        print(f"フレーム読み込み失敗 at {start_frame}")
        cap.release()
        return

    if flip:
        frame = cv2.flip(frame, 0)

    if start_frame == end_frame:
        # グレースケール画像として保存
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_name = f"{name}_frame_{start_frame}.bmp"
        save_path = os.path.join(dir_name, out_name)
        cv2.imwrite(save_path, gray)
        print(f"保存完了: {save_path} （1フレームをグレースケールで保存）")
    else:
        out_name = f"{name}_frames_{start_frame}_{end_frame}.avi"
        save_path = os.path.join(dir_name, out_name)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=True)

        out.write(frame)  # 1枚目

        for current_frame in range(start_frame + 1, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"フレーム読み込み失敗 at {current_frame}")
                break
            if flip:
                frame = cv2.flip(frame, 0)
            out.write(frame)

        out.release()
        print(f"保存完了: {save_path} （{end_frame - start_frame + 1} フレーム）")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="動画からフレームを切り出して保存（オプションで上下反転）")
    parser.add_argument("video", help="入力動画ファイルのパス")
    parser.add_argument("start", type=int, help="開始フレーム番号")
    parser.add_argument("end", type=int, help="終了フレーム番号")
    parser.add_argument("--flip", action="store_true", help="出力を上下反転させる")

    args = parser.parse_args()

    extract_frame_range(args.video, args.start, args.end, args.flip)
