import cv2
import sys

# AVIファイルを読み込み（ファイル名は必要に応じて変更）
path1 = sys.argv[1]
path2 = sys.argv[2]
cap1 = cv2.VideoCapture(path1)
cap2 = cv2.VideoCapture(path2)

# 表示サイズを指定（揃える）
width, height = 640, 360

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # どちらかが再生終了したら終了
    if not ret1 or not ret2:
        break

    # サイズを統一
    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    # 横に並べて結合
    combined = cv2.hconcat([frame1, frame2])

    # 表示
    cv2.imshow('Side-by-Side AVI Playback', combined)

    # 30ms待機：qキーで終了
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 終了処理
cap1.release()
cap2.release()
cv2.destroyAllWindows()
