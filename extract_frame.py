import cv2
import sys
import os

def extract_frame_range(video_path, start_frame, end_frame):
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

    if start_frame == end_frame:
        # グレースケール画像として保存
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_name = f"{name}_frame_{start_frame}.bmp"
        save_path = os.path.join(dir_name, out_name)
        cv2.imwrite(save_path, gray)
        print(f"保存完了: {save_path} （1フレームをグレースケールで保存）")
    else:
        # 複数フレームとして動画で保存
        out_name = f"{name}_frames_{start_frame}_{end_frame}.avi"
        save_path = os.path.join(dir_name, out_name)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=True)

        current_frame = start_frame
        out.write(frame)  # 既に読み込んだ最初のフレームを保存
        current_frame += 1

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"フレーム読み込み失敗 at {current_frame}")
                break
            out.write(frame)
            current_frame += 1

        out.release()
        print(f"保存完了: {save_path} （{end_frame - start_frame + 1} フレーム）")

    cap.release()

def main():
    if len(sys.argv) != 4:
        print("使い方: python extract_avi_range.py input_video.avi start_frame end_frame")
        return

    video_path = sys.argv[1]
    try:
        start_frame = int(sys.argv[2])
        end_frame = int(sys.argv[3])
    except ValueError:
        print("エラー: フレーム番号は整数で指定してください。")
        return

    extract_frame_range(video_path, start_frame, end_frame)

if __name__ == "__main__":
    main()
