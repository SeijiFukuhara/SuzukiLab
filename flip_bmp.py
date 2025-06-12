import cv2
import sys
import os

def flip_and_save(image_path):
    # グレースケール画像として読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"エラー: ファイルが読み込めませんでした -> {image_path}")
        return

    # 上下反転（Y軸反転）
    flipped = cv2.flip(img, 0)

    # 出力ファイル名の生成
    dir_name, base_name = os.path.split(image_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"flip_vertically_{name}{ext}"  # 先頭に付与
    save_path = os.path.join(dir_name, new_name)

    # 画像の保存
    success = cv2.imwrite(save_path, flipped)
    if success:
        print(f"保存完了: {save_path}")
    else:
        print(f"エラー: 保存に失敗しました -> {save_path}")

def main():
    args = sys.argv[1:]  # 最初の要素はスクリプト名なので除外

    if len(args) not in [1, 2]:
        print("エラー: 画像ファイルは1つまたは2つ指定してください。")
        print("使い方: python flip_image.py image1.jpg [image2.bmp]")
        return

    for image_path in args:
        flip_and_save(image_path)

if __name__ == "__main__":
    main()
