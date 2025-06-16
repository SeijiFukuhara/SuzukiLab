import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import signal
from PIL import Image
import os
import re
import string
import sys
import cv2

def loadtext(fname):
    fname_load = np.loadtxt(fname, delimiter = ",")
    return fname_load

def plot_phase(np_array,d_temp):
    fig = plt.figure()
    plt.imshow(np_array, cmap="rainbow")
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label( "Corrected Phase [rad]", fontsize=14)
    plt.clim(0,3.0)
    scalebar = ScaleBar(1/d_temp,'um', location = "lower right", length_fraction = 0.2, font_properties={"size": 20}) #*字大きい，位置違う
    #scalebar = ScaleBar(1/d,'um', location = "upper left") #*1 pixel = ? um もともとの設定
    plt.gca().add_artist(scalebar)
    # figname = fname.replace('.csv', '.png') #*保存先のパス．元データのcsvファイルと全く同じファイル名で保存する設定．
    # plt.savefig(figname)
    # plt.show()

def offset(twolist_array,convolve,convolve_size_temp,z1,z2,x1,x2): #*水温と室温が一致する範囲を指定し，オフセット
    #*以下の段落をコメントアウトしているときは[from scipy import signal]の行に「アクセスできません」というメッセージが表示されるが問題ない
    #TODO 移動平均とる場合は以下最初の空行までを有効にする
    # xxに対してsize個での移動平均を取る
    def valid_convolve(xx, size):
        b = np.ones(size)/size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(size/2)
        # 補正部分
        xx_mean[0] *= size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= size/(i+n_conv)
            xx_mean[-i] *= size/(i + n_conv - (size % 2))
        # size%2は奇数偶数での違いに対応するため
        return xx_mean
    #?自分で書いたけど意味わからない
    if convolve == True:
        twoarray_convolve = []
        onelist_array = []
        for i in range(len(twolist_array)):
            onelist_array = list(valid_convolve(twolist_array[i], convolve_size_temp))
            twoarray_convolve.append(onelist_array)
        twolist_array = np.array(twoarray_convolve)
    #print(twolist_array)
    #TODO 左半分の領域が対象ならコメントアウト
    #img_phase = np.fliplr(img_phase)
    #* [z1:z2,x1:x2]の範囲の温度を平均し，その位相を0にoffset，絶対水温の領域を指定．zは縦方向，xは横方向．順番に注意．
    offset = twolist_array[z1:z2, x1:x2]
    #TODO 位相差の逆転を解消．0次光=ピンホール，1次光=スリットのとき有効にする
    twolist_array = offset.mean() - twolist_array
    return twolist_array

def plot_phase_and_save(np_array, d_temp, fname_path, dir_bmp):
    # path_png = fname_path.replace('.csv', '.png')
    path_bmp = fname_path.replace('.csv', '.bmp')
    
    # filename_png = os.path.basename(path_png)  
    filename_bmp = os.path.basename(path_bmp)  
    
    # save_path_png = os.path.join(dir_png, filename_png)
    save_path_bmp = os.path.join(dir_bmp, filename_bmp)

    fig = plt.figure()
    plt.imshow(np_array, cmap="rainbow")
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label("Corrected Phase [rad]", fontsize=14)
    plt.clim(0, 3.0)
    scalebar = ScaleBar(1/d_temp, 'um', location="lower right",
                        length_fraction=0.2, font_properties={"size": 20})
    plt.gca().add_artist(scalebar)

    #* PNG形式で一時保存してからBMPに変換
    temp_path_png = save_path_bmp.replace('.bmp', '_temp.png')
    plt.savefig(temp_path_png, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # メモリ開放
    
    # PNG → BMP 変換
    img = Image.open(temp_path_png)
    img.save(save_path_bmp)

    # 一時PNGを削除
    os.remove(temp_path_png)

def extract_frame_range_suffix(path):
    """
    拡張子の直前にある 'frames_XXX_YYY' から XXX, YYY を整数として取得
    例: 'sample_video_flip_vertically_frames_420_450.avi' → (420, 450)
    """
    base = os.path.basename(path)
    match = re.search(r'frames_(\d+)_(\d+)', base)
    if not match:
        raise ValueError("ファイル名に 'frames_XXX_YYY' の形式が含まれていません")
    start, end = map(int, match.groups())
    return start, end

def add_tilde_to_filename(src_path, prefix):
    """
    入力パスのファイル名の先頭に '~' を付けた新しいフルパスを返す。

    例:
        src_path = "C:/data/sample.avi"
        → "C:/data/~sample.avi"
    """
    dir_name = os.path.dirname(src_path)
    base_name = os.path.basename(src_path)
    new_base_name = prefix + base_name
    outpath = os.path.join(dir_name, new_base_name)
    return outpath

def find_available_filename(input_path):
    """
    base_dir: 探索対象のディレクトリ
    base_filename: ベースファイル名（例: 'result.txt'）
    
    戻り値: 利用可能な 'a_result.txt' ～ 'z_result.txt' のうち最初の未使用名
    """
    base_dir, base_filename = os.path.split(input_path)
    existing_names = set(os.listdir(base_dir))
    for prefix in string.ascii_lowercase:  # 'a' から 'z' まで
        candidate_prefix = f"~{prefix}_"
        # 同名のファイル or フォルダが存在するかを「名前の先頭一致」で確認
        if not any(name.startswith(candidate_prefix) for name in existing_names):
            return candidate_prefix
    raise FileExistsError("a_〜z_まで全てのファイル名が既に存在しています。")

def find_available_filename_combination(input_path):
    """
    入力パスの basename（末尾）に ~a_ のような形式が含まれているかをチェックし、
    ~a1_ ～ ~a9_ の中で未使用のプレフィックスを返す。
    """
    base_dir = os.path.dirname(input_path)
    tail_name = os.path.basename(input_path)  # ← ← 最後のフォルダ名またはファイル名を対象とする

    # tail_name に ~a_ のような形式が含まれているかをチェック
    match = re.match(r"~([a-z])_", tail_name)
    if not match:
        raise ValueError(f"パス末尾に ~a_ ～ ~z_ の形式が含まれていません: {tail_name}")

    letter = match.group(1)
    existing_names = set(os.listdir(base_dir))

    for number in range(1, 10):  # ~a1_ ～ ~a9_
        candidate_prefix = f"~{letter}{number}_"
        if not any(name.startswith(candidate_prefix) for name in existing_names):
            return candidate_prefix

    raise FileExistsError(f"{tail_name} に対する ~{letter}1_ 〜 ~{letter}9_ がすべて使用されています。")

def video2images(video): 
    if not __HAS_OPENCV__:
        raise ImportError(
            "Open cv must be installed to read video.\n"
        )

    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read() #? 読み込み成功のブール値、読み込んだ画像データ（Numpy配列）
    count = 0
    images = []
    
    while success:
        success, image = vidcap.read()
        image = np.array(image)
        # here we just use the r channel. Maybe we need something here
        #? コメントではrチャンネル（RGB配列の0番目を取り出しているから）を使っていると書いているが、BGR配列の可能性もある。
        #? これはカラー画像をから一つのチャンネルのみを取り出すことで疑似的なグレースケールを創り出す操作をしている。
        if image.ndim == 3:
            image = image[..., 0]
        if success:
            images.append(image)

    return images[0], images

