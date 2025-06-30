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
from scipy.optimize import curve_fit
import traceback

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

def offset(twolist_array,convolve_size_temp,z1,z2,x1,x2,convolve): #*水温と室温が一致する範囲を指定し，オフセット
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

#! extract_phse,py中の「def _video2images(video):」は入力が100フレームの時、0～n-2フレーム目（そうフレーム数n-1）までしか出力されない。（意図的かは不明）なので、出力フレーム数（imagesのフレーム数）がn枚になるように新たに関数を定義する。
def video2images_rewrite(video): 

    vidcap = cv2.VideoCapture(video)
    images = []

    while True:
        success, image = vidcap.read()
        if not success:
            break
        image = np.array(image)
        if image.ndim == 3:
            image = image[..., 0]
        images.append(image)

    return images

def _video2images(vidcap):

    # vidcap = cv2.VideoCapture(video)
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

import cv2

def load_video_with_leading_image(image_path, video_path):
    """
    .bmp画像を先頭に挿入した動画フレームのリストを返す。
    動画ファイルや中間ファイルは保存しない。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("動画が開けません")

    # 参照画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise IOError("画像が読み込めません")

    # 動画と画像のサイズが一致するかチェック
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if image.shape[:2] != (height, width):
        raise ValueError(f"画像サイズが動画と一致しません。画像: {image.shape[:2]}, 動画: {(height, width)}")

    # 結果格納用リスト
    frames = [image]

    # 動画フレームをすべて読み込む
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames  # ← NumPy配列のリスト



###############################################################################################################
#! 温度分布
#*位相のカラーマップを表示する関数
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
    return fig

# def y0(x,popt):
#     y0 = popt[0]*np.exp(-popt[2]*(x - popt[1])**2) + x*popt[3] + popt[4]
#     return y0

# def y1(x,popt):
#     y1 = popt[0]*np.exp(-popt[2]*(x - popt[1])**2)
#     return y1

# def y2(x,popt):
#     y2 = popt[0]*np.exp(-popt[2]*x**2)
#     return y2

# def y3(x,popt):
#     y3 = -popt[0]*np.exp(-popt[2]*x**2)
#     return y3

def  gaussian_plus_linear(x, A, mu, sigma, m, b):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + m * x + b

def  gaussian_plus_offset(x, A, mu, sigma, b):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + b

def  gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def  gaussian_centered(x, A, sigma):
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2))

def negative_gaussian_centered(x, A, sigma):
    return -A * np.exp(-(x ** 2) / (2 * sigma ** 2))

def estimate_initial_gaussian_params(x, y):
    # A_init = np.max(y)
    A_init = np.max(y) - np.min(y)

    threshold = 0.5 * np.max(y)
    mask = y > threshold
    if np.any(mask):
        mu_init = np.sum(x[mask] * y[mask]) / np.sum(y[mask])

    half_max = np.max(y) / 2
    indices = np.where(y >= half_max)[0]
    if len(indices) >= 2:
        fwhm = x[indices[-1]] - x[indices[0]]
        sigma_init = fwhm / 2.355
    else:
        sigma_init = (np.max(x) - np.min(x)) / 4  # fallback
    sigma_init = max(sigma_init, 1e-6)

    m_init = 0.0001  # 初期値

    b_init = 0.15  # 初期値
    return A_init, mu_init, sigma_init, m_init, b_init

def approximation_phase(twolist_array, x_axis, width_phase,height_phase,n_apr_pix, gaussian_additive_term):
    # popt_full = np.array([]).reshape(0, 5)  # (0, 5) の空配列

    if gaussian_additive_term == "linear":
        n_params = 5
        func = gaussian_plus_linear
        popt_full = np.zeros((n_apr_pix, n_params))
        popt_init_full = np.zeros((n_apr_pix, n_params))
        # y0, y1, y2, y3 をリストに格納
        y_functions = [
                        gaussian_plus_linear,
                        gaussian,
                        gaussian_centered,
                        negative_gaussian_centered
                        ]

        num_functions = len(y_functions)
        phase_apr = np.zeros((num_functions, n_apr_pix, width_phase))  # 4つのフェーズデータをまとめて作成

        for i in range(n_apr_pix):
            array_y = np.array(twolist_array[i])
            #* 初期パラメータの推定
            A_init, mu_init, sigma_init, m_init, b_init = estimate_initial_gaussian_params(x_axis, array_y)
            #* フィッティング（ガウシアン+ 一次関数）
            popt, _ = curve_fit(func, x_axis, array_y, maxfev=20000,
                                p0=[A_init, mu_init, sigma_init, m_init, b_init],
                                bounds=([A_init - 0.5, mu_init - 20, sigma_init - 15, m_init - 0.1, b_init - 1],
                                        [A_init + 0.5, mu_init + 20, sigma_init + 15, m_init + 0.1, b_init + 1])
                                )
            popt_init_full[i] = [A_init, mu_init, sigma_init, m_init, b_init]
            popt_full[i] = popt  # フィッティング結果を格納

        # 各フェーズデータを計算し、対応する配列に格納
        for i, y_func in enumerate(y_functions):
            for j in range(n_apr_pix):
                A, mu, sigma, m, b = popt_full[j]
                if y_func == gaussian_plus_linear:
                    y_values = gaussian_plus_linear(x_axis, A, mu, sigma, m, b)
                elif y_func == gaussian:
                    y_values = gaussian(x_axis, A, mu, sigma)
                elif y_func == gaussian_centered:
                    y_values = gaussian_centered(x_axis, A, sigma)
                elif y_func == negative_gaussian_centered:
                    y_values = negative_gaussian_centered(x_axis, A, sigma)
                phase_apr[i, j, :] = y_values  # 横方向に格納

    elif gaussian_additive_term == "constant":
        n_params = 4
        func = gaussian_plus_offset
        popt_full = np.zeros((n_apr_pix, n_params))
        popt_init_full = np.zeros((n_apr_pix, n_params))
        # y0, y1, y2, y3 をリストに格納
        y_functions = [
                        gaussian_plus_offset,
                        gaussian,
                        gaussian_centered,
                        negative_gaussian_centered
                        ]

        num_functions = len(y_functions)
        phase_apr = np.zeros((num_functions, n_apr_pix, width_phase))  # 4つのフェーズデータをまとめて作成

        for i in range(n_apr_pix):
            array_y = np.array(twolist_array[i])
            #* 初期パラメータの推定
            A_init, mu_init, sigma_init, _, b_init = estimate_initial_gaussian_params(x_axis, array_y)
            #* フィッティング（ガウシアン+ 定数）
            popt, _ = curve_fit(func, x_axis, array_y, maxfev=20000,
                                p0=[A_init, mu_init, sigma_init, b_init],
                                bounds=([A_init - 0.5, mu_init - 20, sigma_init - 15, b_init - 1],
                                        [A_init + 0.5, mu_init + 20, sigma_init + 15, b_init + 1])
                                )
            popt_init_full[i] = [A_init, mu_init, sigma_init, b_init]
            popt_full[i] = popt  # フィッティング結果を格納

        # 各フェーズデータを計算し、対応する配列に格納
        for i, y_func in enumerate(y_functions):
            phase_list = []
            for j in range(n_apr_pix):
                A, mu, sigma, b = popt_full[j]
                if y_func == gaussian_plus_offset:
                    y_values = gaussian_plus_offset(x_axis, A, mu, sigma, b)
                elif y_func == gaussian:
                    y_values = gaussian(x_axis, A, mu, sigma)
                elif y_func == gaussian_centered:
                    y_values = gaussian_centered(x_axis, A, sigma)
                elif y_func == negative_gaussian_centered:
                    y_values = negative_gaussian_centered(x_axis, A, sigma)
                phase_apr[i, j, :] = y_values  # 横方向に格納
    else:
        raise ValueError("gaussian_additive_term must be 'linear' or 'constant'")

    twolist_expanded = twolist_array[np.newaxis, :, :]
    phase_full = np.tile(twolist_expanded, (num_functions, 1, 1))
    for i in range(num_functions):
        phase_full[i][:phase_apr[i].shape[0], :] = phase_apr[i]

    return phase_full, popt_full, popt_init_full




