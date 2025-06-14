import cv2
import sys
import os
import glob

# フォルダ指定（コマンドライン引数）
folder1 = sys.argv[1]
folder2 = sys.argv[2]

# BMPファイルの一覧を取得
files1 = sorted(glob.glob(os.path.join(folder1, "*.bmp")))
files2 = sorted(glob.glob(os.path.join(folder2, "*.bmp")))
num_frames = min(len(files1), len(files2))

if num_frames == 0:
    print("画像が見つかりません")
    sys.exit(1)

# サイズ設定
resize_width, resize_height = 640, 480
paused = False
frame_idx = 0
seek_request = False

# コールバック関数（シークバー移動時）
def on_trackbar(val):
    global frame_idx, paused, seek_request
    frame_idx = val
    paused = True
    seek_request = True  # バー移動時は即座にフレーム更新

# ウィンドウとトラックバーを作成
cv2.namedWindow("Player")
cv2.createTrackbar("Position", "Player", 0, num_frames - 1, on_trackbar)

while True:
    if not paused or seek_request:
        img1 = cv2.imread(files1[frame_idx])
        img2 = cv2.imread(files2[frame_idx])

        if img1 is None or img2 is None:
            print(f"読み込み失敗: {files1[frame_idx]} または {files2[frame_idx]}")
            frame_idx = (frame_idx + 1) % num_frames
            continue

        img1 = cv2.resize(img1, (resize_width, resize_height))
        img2 = cv2.resize(img2, (resize_width, resize_height))
        combined = cv2.hconcat([img1, img2])

        cv2.imshow("Player", combined)
        cv2.setTrackbarPos("Position", "Player", frame_idx)

        # seek 直後は再生停止に戻す
        seek_request = False

        if not paused:
            frame_idx = (frame_idx + 1) % num_frames  # 繰り返し再生

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

cv2.destroyAllWindows()
