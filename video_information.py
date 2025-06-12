# 動画ファイルの情報を取得するスクリプト
import cv2

import sys

video_path = sys.argv[1]
vidcap = cv2.VideoCapture(video_path)

if not vidcap.isOpened():
    print("動画ファイルを開けませんでした。")
else:
    # フレーム基本情報
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # コーデック（FourCC）情報
    fourcc = int(vidcap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print("🔹 動画ファイル情報")
    print(f"・総フレーム数     : {frame_count}")
    print(f"・フレームレート   : {fps:.2f} fps")
    print(f"・フレームサイズ   : {width} x {height}")
    print(f"・コーデック       : {codec} (FourCC)")

    # 最初のフレームで画像配列の形式確認
    success, image = vidcap.read()
    if success:
        print("\n🔹 最初のフレーム情報")
        print(f"・image.shape       : {image.shape}")
        print(f"・image.dtype       : {image.dtype}")
        print(f"・次元数（ndim）    : {image.ndim}")

        if image.ndim == 3:
            ch = image.shape[2]
            if ch == 3:
                print("・チャンネル構成    : カラー（BGR）")
            elif ch == 4:
                print("・チャンネル構成    : カラー＋アルファ（BGRA）")
            else:
                print(f"・チャンネル構成    : 不明な3次元カラー画像（チャンネル数: {ch}）")
        elif image.ndim == 2:
            print("・チャンネル構成    : グレースケール")
        else:
            print("・チャンネル構成    : 不明な形式")
    else:
        print("フレームの読み込みに失敗しました。")

    vidcap.release()
