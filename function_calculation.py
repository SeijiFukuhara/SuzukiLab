import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import signal
from PIL import Image
import os
import re

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

def offset(twolist_array,convolve_mode,convolve_size_temp,z1,z2,x1,x2): #*水温と室温が一致する範囲を指定し，オフセット
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
    twoarray = []
    onelist_array = []
    for i in range(len(twolist_array)):
        onelist_array = list(valid_convolve(twolist_array[i], convolve_size_temp))
        twoarray.append(onelist_array)
    #?自分で書いたけど意味わからない
    if convolve_mode == 0:
        twolist_array = np.array(twoarray)
    #print(twolist_array)
    #TODO 左半分の領域が対象ならコメントアウト
    #img_phase = np.fliplr(img_phase)
    #* [z1:z2,x1:x2]の範囲の温度を平均し，その位相を0にoffset，絶対水温の領域を指定．zは縦方向，xは横方向．順番に注意．
    offset = twolist_array[z1:z2, x1:x2]
    #TODO 位相差の逆転を解消．0次光=ピンホール，1次光=スリットのとき有効にする
    twolist_array = offset.mean() - twolist_array
    return twolist_array

def plot_phase_and_save(np_array, d_temp, fname_path, dir_bmp, dir_png):
    path_png = fname_path.replace('.csv', '.png')
    path_bmp = fname_path.replace('.csv', '.bmp')
    
    filename_png = os.path.basename(path_png)  
    filename_bmp = os.path.basename(path_bmp)  
    
    save_path_png = os.path.join(dir_png, filename_png)
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

    # 保存（拡張子は自動で適用されます：.bmp）
    plt.savefig(save_path_png, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # メモリ開放
    
    # PNG → BMP 変換
    img = Image.open(save_path_png)
    img.save(save_path_bmp)

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
