import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import signal
from PIL import Image
import os
import re
import string
import cv2
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.linalg import lu_factor, lu_solve
from functools import partial

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
#! 位相分布
import numpy as np

def shift_array(array, a, fill_value=0):
    """
    array: 1D numpy array
    a: int, シフト量（正で左シフト、負で右シフト）
    fill_value: シフトで空いた部分に入れる値
    """
    shifted = np.full_like(array, fill_value)

    if a > 0:
        # 左にシフト
        shifted[:-a] = array[a:]
    elif a < 0:
        # 右にシフト
        shifted[-a:] = array[:a]
    else:
        shifted[:] = array  # シフトなし

    return shifted

def shift_2d_array(array_2d, shift_list, fill_value=0):
    """
    array_2d: 2D numpy array
    shift_list: 各行に対するシフト量（長さはarray_2dの行数以下）
    fill_value: 埋める値
    """
    shifted_array_2d = np.full_like(array_2d, fill_value)
    num_rows = array_2d.shape[0]

    if len(shift_list) > num_rows:
        raise ValueError(f"shift_list の長さ ({len(shift_list)}) が array_2d の行数 ({num_rows}) を超えています。")

    for i in range(len(shift_list)):
        shifted_array_2d[i] = shift_array(array_2d[i], shift_list[i], fill_value)

    # shift_list が短い場合、残りの行はそのままコピー
    for i in range(len(shift_list), num_rows):
        shifted_array_2d[i] = array_2d[i]

    return shifted_array_2d



#*位相のカラーマップを表示する関数
# def plot_phase(np_array,d_temp):
#     fig = plt.figure()
#     plt.imshow(np_array, cmap="rainbow")
#     plt.axis('off')
#     cbar = plt.colorbar()
#     cbar.set_label( "Corrected Phase [rad]", fontsize=14)
#     plt.clim(0,3.0)
#     scalebar = ScaleBar(1/d_temp,'um', location = "lower right", length_fraction = 0.2, font_properties={"size": 20}) #*字大きい，位置違う
#     #scalebar = ScaleBar(1/d,'um', location = "upper left") #*1 pixel = ? um もともとの設定
#     plt.gca().add_artist(scalebar)
#     # figname = fname.replace('.csv', '.png') #*保存先のパス．元データのcsvファイルと全く同じファイル名で保存する設定．
#     # plt.savefig(figname)
#     return fig

def plot_phase(np_array, d_temp):
    fig, ax = plt.subplots()  # fig, ax を生成
    im = ax.imshow(np_array, cmap="rainbow")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Corrected Phase [rad]", fontsize=14)
    im.set_clim(0, 3.0)

    scalebar = ScaleBar(1/d_temp, 'um', location="lower right", length_fraction=0.2, font_properties={"size": 20})
    ax.add_artist(scalebar)

    return fig



def  gaussian_plus_linear(x, A, mu, sigma, m, b):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + (m * 10**(-4)) * x + b

def  gaussian_plus_linear_centered(x, A, sigma, m, b):
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2)) + (m * 10**(-4)) * x + b

def  gaussian_plus_offset(x, A, mu, sigma, b):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + b

def  gaussian_plus_offset_centered(x, A,sigma, b):
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2)) + b

def  gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def  gaussian_centered(x, A, sigma):
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2))

def  gaussian_plus_parabola_centered(x, A, sigma, a, c):
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2)) + a * x**2 + c

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

    n_edge = 10  # 両端n点ずつ
    x_edge = np.concatenate([x[:n_edge], x[-n_edge:]])
    y_edge = np.concatenate([y[:n_edge], y[-n_edge:]])

    m_init, b_init, _, _, _ = linregress(x_edge, y_edge)
    m_init = m_init * 10**4

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
                        gaussian_plus_linear_centered,
                        gaussian_plus_offset,
                        gaussian_plus_offset_centered,
                        # gaussian,
                        # gaussian_centered,
                        ]
        popt_names = ["A", "mu", "sigma", "m", "b"]
        popt_init_names = ["A_init", "mu_init", "sigma_init", "m_init", "b_init"]

        num_functions = len(y_functions)
        phase_apr = np.zeros((num_functions, n_apr_pix, width_phase))  # 4つのフェーズデータをまとめて作成

        for i in range(n_apr_pix):
            array_y = np.array(twolist_array[i])
            #* 初期パラメータの推定
            A_init, mu_init, sigma_init, m_init, b_init = estimate_initial_gaussian_params(x_axis, array_y)
            #* フィッティング（ガウシアン+ 一次関数）
            popt, _ = curve_fit(func, x_axis, array_y, maxfev=20000,
                                p0=[A_init, mu_init, sigma_init, m_init, b_init],
                                bounds=([A_init - 0.5, mu_init - 20, sigma_init - 15, m_init - 1, b_init - 1],
                                        [A_init + 0.5, mu_init + 20, sigma_init + 15, m_init + 1, b_init + 1])
                                )
            popt_init_full[i] = [A_init, mu_init, sigma_init, m_init, b_init]
            popt_full[i] = popt  # フィッティング結果を格納

        # 各フェーズデータを計算し、対応する配列に格納
        for i, y_func in enumerate(y_functions):
            for j in range(n_apr_pix):
                A, mu, sigma, m, b = popt_full[j]
                if y_func == gaussian_plus_linear:
                    y_values = gaussian_plus_linear(x_axis, A, mu, sigma, m, b)
                elif y_func == gaussian_plus_linear_centered:
                    y_values = gaussian_plus_linear_centered(x_axis, A, sigma,m, b)
                elif y_func == gaussian_plus_offset:
                    y_values = gaussian_plus_offset(x_axis, A, mu, sigma, b)
                elif y_func == gaussian_plus_offset_centered:
                    y_values = gaussian_plus_offset_centered(x_axis, A,  sigma, b)
                elif y_func == gaussian:
                    y_values = gaussian(x_axis, A, mu, sigma)
                elif y_func == gaussian_centered:
                    y_values = gaussian_centered(x_axis, A, sigma)
                phase_apr[i, j, :] = y_values  # 横方向に格納

        popt_init_full[:, 3] = popt_init_full[:, 3] * (10**(-4))  # m_initを10^(-4)倍
        popt_full[:, 3] = popt_full[:, 3] * (10**(-4))  # m_initを10^(-4)倍

        twolist_expanded = twolist_array[np.newaxis, :, :]
        phase_full = np.tile(twolist_expanded, (num_functions, 1, 1))
        for i in range(num_functions):
            phase_full[i][:phase_apr[i].shape[0], :] = phase_apr[i]


    elif gaussian_additive_term == "constant":
        n_params = 4
        func = gaussian_plus_offset
        popt_full = np.zeros((n_apr_pix, n_params))
        popt_init_full = np.zeros((n_apr_pix, n_params))
        # y0, y1, y2, y3 をリストに格納
        y_functions = [
                        gaussian_plus_offset,
                        gaussian_plus_offset_centered,
                        # gaussian,
                        # gaussian_centered,
                        ]

        popt_names = ["A", "mu", "sigma", "b"]
        popt_init_names = ["A_init", "mu_init", "sigma_init", "b_init"]

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
                elif y_func == gaussian_plus_offset_centered:
                    y_values = gaussian_plus_offset_centered(x_axis, A,sigma, b)
                elif y_func == gaussian:
                    y_values = gaussian(x_axis, A, mu, sigma)
                elif y_func == gaussian_centered:
                    y_values = gaussian_centered(x_axis, A, sigma)

                phase_apr[i, j, :] = y_values  # 横方向に格納

        twolist_expanded = twolist_array[np.newaxis, :, :]
        phase_full = np.tile(twolist_expanded, (num_functions, 1, 1))
        for i in range(num_functions):
            phase_full[i][:phase_apr[i].shape[0], :] = phase_apr[i]

    else:

        raise ValueError("gaussian_additive_term must be 'linear' or 'constant'")

    # phase_apr_dict = dict(zip(y_functions, phase_full))
    phase_apr_dict = {func.__name__: value for func, value in zip(y_functions, phase_full)}
    popt_dict = dict(zip(popt_names, popt_full.T))
    popt_init_dict = dict(zip(popt_init_names, popt_init_full.T))

    return phase_apr_dict, popt_dict, popt_init_dict


###############################################################################################################
#! 温度分布
def refractive(t):#*参考論文より近似して得た，温度(t)と水の屈折率(ref)の関係．屈折率は温度に関する二次方程式で表される．
    ref = - 0.00000113*t*t - 0.00005285*t + 1.33758
    return ref

def solve_T(a, b, c): #*ref=0とした時の上の方程式を解く．a*x**2 + b*x + c = 0 の解
    D = np.sqrt(abs(b**2 - 4*a*c))
    T = (-b + D) / (2 * a)
    return T

def calc_temp(twolist_array, x_axis, Nx, Nz, mode, l, d_temp, n_room, lamda):
    # img_phase_array_slice = twolist_array[:,l:Nx+1] #TODO 左半分
    img_phase_array_slice = twolist_array[:,Nx:-l] #TODO 右半分

    # x_axis_half = x_axis[l:Nx+1] #TODO 左半分
    x_axis_half = x_axis[Nx:-l] #TODO 右半分

    Nx = len(img_phase_array_slice[0]) #? Nxの再定義（ややこしい）
    # img_phase_array_slice_flip = np.fliplr(img_phase_array_slice) #TODO 左半分
    img_phase_array_slice_flip = img_phase_array_slice #TODO 右半分

    T_solution = np.zeros((Nz,Nx)) #*温度結果格納用のリスト．1つのリスト内に要素がNx個入っているリストをNz個並べ，リストにする．np.zerosによって要素の値はすべて0
    ref_solution = np.zeros((Nz,Nx)) #*屈折率格納用のリスト．中身は上と全く同じ

    #*多次元(今回は2次元)配列を作る
    #TODO メッシュサイズ，変更する可能性あり
    if mode == 0:
        mesh = np.array([[40, 70, 100, 130, 160, Nx],
                        [ 1,  2,   4,   6,   8,  10]])
    #*mesh1/数時間かかる？
    elif mode == 1:
        mesh = np.array([[Nx],
                        [1]])
    #*mesh2/十数分かかる
    elif mode == 2:
        mesh = np.array([[250, Nx],
                        [ 1, 10]])
    #*mesh3/
    elif mode == 3:
        mesh = np.array([[160, Nx],
                        [ 1, 10]])
    else:
        print("meshmodeの入力が間違っています")
    #*print(r)を見れば何をしているかわかる
    r = np.array([0]) #*空のベクトルを用意
    for i,j in enumerate(mesh[0,:]): #*0番目のリストを対象に，iは要素のインデックス，jは要素そのもの
        while r[-1] < j: #*r[-1](rの最後の要素)がjより小さい間は
            r = np.append(r, r[-1]+(mesh[1,i]))

    #*rの最後の要素を削除して，Nxを追加
    r = np.delete(r, -1)
    r[-1] = Nx

    #*An = b
    #*r.size-1 行，r*size-1 列の上三角正方行列を作る
    A = np.zeros((r.size-1, r.size-1))
    for i in range(r.size-1):
        for j in range(r.size-1):
            if i > j: #*iがjより大きいときは0（下三角成分）
                A[i,j] = 0
            else : #*iがj以下のときは（対角成分含めて）計算（上三角成分） d_temp[pix_μm]
                A[i,j] = 2 * np.sqrt((r[j+1]/d_temp)**2 - (r[i]/d_temp)**2) - 2 * np.sqrt((r[j]/d_temp)**2 - (r[i]/d_temp)**2)

    #print(A)
    for k in range(Nz): #*あるz＝一定(ピクセル単位)の平面でTを求めるのを繰り返す
        b = np.zeros(r.size - 1)
        for i in range(r.size-1):
            b[i] = 2 * n_room * np.sqrt((Nx/d_temp)**2 - (r[i]/d_temp)**2) - img_phase_array_slice_flip[k,r[i]:r[i+1]].mean() * lamda / (2*np.pi)
            #b[i] = 2 * n_room * np.sqrt((Nx/d)**2 - (r[i]/d)**2) - (img_phase[k][r[i]:r[i+1]].mean() - phase_edge) * lamda / (2*np.pi)
        lu_solution = lu_solve(lu_factor(A), b) #*屈折率分布　An = bを解く

        p = 0     #*出てきた解を画像の大きさと一緒にするため
        for i in range(r.size-1):
            while p < r[i+1]:
                ref_solution[k,p] = lu_solution[i]
                p += 1
        ref_solution[k, -1] = lu_solution[-1]

    #*(屈折率)=(温度に関する二次方程式)の関係式に３つの引数を代入
        T_solution[k, :] = solve_T(0.00000113, 0.00005285, ref_solution[k, :] - 1.33758)

    T_solution[np.isnan(T_solution)] = True  #*solve_Tで出た複素解はnanになるので，boolで置換とりあえず
    #TODO 左半分の場合は有効
    # T_solution = np.fliplr(T_solution)
    return T_solution, r, x_axis_half

def plot_temp(np_array, d_temp):
    fig, ax = plt.subplots()  # fig, ax を生成
    im = ax.imshow(np_array, cmap="rainbow")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, label="Temperature [℃]")
    im.set_clim(23, 60)

    scalebar = ScaleBar(1/d_temp, 'um', length_fraction=0.3, location="upper left")
    ax.add_artist(scalebar)

    return fig


###############################################################################################################
#! 熱流束
def cut_array_by_threshold(array, target, threshold, from_end=False):
    if from_end:
        indices = range(len(array)-1, -1, -1)  # 末尾→先頭
    else:
        indices = range(len(array))  # 先頭→末尾

    for i in indices:
        diff = array[i] - target
        if diff < threshold:
            # i番目でカット
            if from_end:
                return array[i+1:]
            else:
                return array[:i]

    # 条件を満たさない場合
    return array

def cut_2d_array_by_threshold(array_2d, target, threshold, from_end=False):
    """
    二次元配列(array_2d)の各行（一次元配列）にcut_array_by_thresholdを適用し、
    結果を二次元配列として返す。
    """
    result = []
    for row in array_2d:
        cut_row = cut_array_by_threshold(row, target, threshold, from_end)
        result.append(cut_row)
    return result

def estimate_initial_gaussian_params_cutoff_temp(x, y):
    """
    x: 1D array, x data
    y: 1D array, y data

    戻り値: A_init, mu_init, sigma_init, b_init
    """

    # 振幅初期値: 最大値 - オフセット
    A_init = np.max(y)

    # 中心位置初期値: 最大値のx位置
    # mu_init = x[np.argmax(y)]

    # σ初期値: 分散の重み付き平均で推定
    y_adj = y
    y_adj[y_adj < 0] = 0  # 負値はゼロにする

    if np.sum(y_adj) == 0:
        sigma_init = 1.0  # 適当な値
    else:
        mu_weighted = np.sum(x * y_adj) / np.sum(y_adj)
        sigma_init = np.sqrt(np.sum(y_adj * (x - mu_weighted) ** 2) / np.sum(y_adj))
    return A_init, sigma_init

def estimate_initial_gaussian_params_cutoff_flow(x, y):
    """
    x: 1D array, x data
    y: 1D array, y data

    戻り値: A_init, mu_init, sigma_init, b_init
    """

    # オフセット初期値: 最小値（背景）
    b_init = np.min(y)

    # 振幅初期値: 最大値 - オフセット
    A_init = np.max(y) - b_init

    # 中心位置初期値: 最大値のx位置
    mu_init = x[np.argmax(y)]

    # σ初期値: 分散の重み付き平均で推定
    y_adj = y - b_init
    y_adj[y_adj < 0] = 0  # 負値はゼロにする

    if np.sum(y_adj) == 0:
        sigma_init = 1.0  # 適当な値
    else:
        mu_weighted = np.sum(x * y_adj) / np.sum(y_adj)
        sigma_init = np.sqrt(np.sum(y_adj * (x - mu_weighted) ** 2) / np.sum(y_adj))

    return A_init, mu_init, sigma_init, b_init

def approximation_cutoff(twolist_array, twolist_x_axis, n_apr_pix, T_room,  mode):
    if mode == "temp":
        n_params = 2
        func_estimate = estimate_initial_gaussian_params_cutoff_temp
        func_approx = gaussian_centered
        popt_full = np.zeros((n_apr_pix, n_params))
        popt_init_full = np.zeros((n_apr_pix, n_params))

        popt_names = ["A","sigma"]
        popt_init_names = ["A_init", "sigma_init"]

        temp_apr = np.empty(n_apr_pix, dtype=object)

        for i in range(n_apr_pix):
            width_temp = len(twolist_array[i])  # 横方向のピクセル
            temp_apr[i] = np.zeros(width_temp)
            x_axis = twolist_x_axis[i]  # 横方向のピクセル
            array_y = np.array(twolist_array[i]) - T_room
            #* 初期パラメータの推定
            A_init, sigma_init = func_estimate(x_axis, array_y)
            # print("x_axis.shape:", x_axis.shape, "array_y.shape:", array_y.shape)
            
            #* フィッティング（ガウシアン）
            popt, _ = curve_fit(func_approx, x_axis, array_y, maxfev=20000,
                                p0=[A_init, sigma_init],
                                # bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf])
                                )
            # popt_init_full[i] = [A_init, mu_init, sigma_init]
            # popt_full[i] = popt  # フィッティング結果を格納
            # print("A_init:", A_init, "sigma_init:", sigma_init)
            # print("A:", popt[0], "sigma:", popt[1])
            popt_init_full[i] = [A_init, sigma_init]
            popt_full[i] = popt  # フィッティング結果を格納

        # 各フェーズデータを計算し、対応する配列に格納
            A, sigma = popt_full[i]
            y_values = func_approx(x_axis, A, sigma) + T_room
            temp_apr[i] = y_values  # 横方向に格納

        # popt_init_full[:, 3] = popt_init_full[:, 3] * (10**(-4))  # m_initを10^(-4)倍
        # popt_full[:, 3] = popt_full[:, 3] * (10**(-4))  # m_initを10^(-4)倍

        # twolist_expanded = twolist_array[np.newaxis, :, :]
        # phase_full = np.tile(twolist_expanded, (num_functions, 1, 1))
        # for i in range(num_functions):
        #     phase_full[i][:phase_apr[i].shape[0], :] = phase_apr[i]

    elif mode == "flow":
        n_params = 4
        # func = gaussian_plus_offset
        # popt_full = np.zeros((n_apr_pix, n_params))
        # popt_init_full = np.zeros((n_apr_pix, n_params))
        # # y0, y1, y2, y3 をリストに格納
        # y_functions = [
        #                 gaussian_plus_offset,
        #                 # gaussian,
        #                 # gaussian_centered,
        #                 # negative_gaussian_centered
        #                 ]

        # popt_names = ["A", "mu", "sigma", "b"]
        # popt_init_names = ["A_init", "mu_init", "sigma_init", "b_init"]

        # num_functions = len(y_functions)
        # phase_apr = np.zeros((num_functions, n_apr_pix, width_phase))  # 4つのフェーズデータをまとめて作成

        # for i in range(n_apr_pix):
        #     array_y = np.array(twolist_array[i])
        #     #* 初期パラメータの推定
        #     A_init, mu_init, sigma_init, _, b_init = estimate_initial_gaussian_params(x_axis, array_y)
        #     #* フィッティング（ガウシアン+ 定数）
        #     popt, _ = curve_fit(func, x_axis, array_y, maxfev=20000,
        #                         p0=[A_init, mu_init, sigma_init, b_init],
        #                         bounds=([A_init - 0.5, mu_init - 20, sigma_init - 15, b_init - 1],
        #                                 [A_init + 0.5, mu_init + 20, sigma_init + 15, b_init + 1])
        #                         )
        #     popt_init_full[i] = [A_init, mu_init, sigma_init, b_init]
        #     popt_full[i] = popt  # フィッティング結果を格納

        # # 各フェーズデータを計算し、対応する配列に格納
        # for i, y_func in enumerate(y_functions):
        #     phase_list = []
        #     for j in range(n_apr_pix):
        #         A, mu, sigma, b = popt_full[j]
        #         if y_func == gaussian_plus_offset:
        #             y_values = gaussian_plus_offset(x_axis, A, mu, sigma, b)
        #         elif y_func == gaussian:
        #             y_values = gaussian(x_axis, A, mu, sigma)
        #         elif y_func == gaussian_centered:
        #             y_values = gaussian_centered(x_axis, A, sigma)
        #         elif y_func == negative_gaussian_centered:
        #             y_values = negative_gaussian_centered(x_axis, A, sigma, b)
        #         phase_apr[i, j, :] = y_values  # 横方向に格納

        # twolist_expanded = twolist_array[np.newaxis, :, :]
        # phase_full = np.tile(twolist_expanded, (num_functions, 1, 1))
        # for i in range(num_functions):
        #     phase_full[i][:phase_apr[i].shape[0], :] = phase_apr[i]

    else:

        raise ValueError("gaussian_additive_term must be 'linear' or 'constant'")

    # phase_apr_dict = dict(zip(y_functions, phase_full))
    # phase_apr_dict = {func.__name__: value for func, value in zip(y_functions, phase_full)}
    popt_dict = dict(zip(popt_names, popt_full.T))
    popt_init_dict = dict(zip(popt_init_names, popt_init_full.T))

    return temp_apr, popt_dict, popt_init_dict

def approximation_cutoff_temp(twolist_array, twolist_x_axis, n_apr_pix, T_room):
    n_params = 2
    func_estimate = estimate_initial_gaussian_params_cutoff_temp
    func_approx = gaussian_centered
    popt_full = np.zeros((n_apr_pix, n_params))
    popt_init_full = np.zeros((n_apr_pix, n_params))

    popt_names = ["A","sigma"]
    popt_init_names = ["A_init", "sigma_init"]

    temp_apr = np.empty(n_apr_pix, dtype=object)

    for i in range(n_apr_pix):
        width_temp = len(twolist_array[i])  # 横方向のピクセル
        temp_apr[i] = np.zeros(width_temp)
        x_axis = twolist_x_axis[i]  # 横方向のピクセル
        array_y = np.array(twolist_array[i]) - T_room
        #* 初期パラメータの推定
        A_init, sigma_init = func_estimate(x_axis, array_y)
        # print("x_axis.shape:", x_axis.shape, "array_y.shape:", array_y.shape)

        #* フィッティング（ガウシアン）
        popt, _ = curve_fit(func_approx, x_axis, array_y, maxfev=20000,
                            p0=[A_init, sigma_init],
                            # bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf])
                            )
        # popt_init_full[i] = [A_init, mu_init, sigma_init]
        # popt_full[i] = popt  # フィッティング結果を格納
        # print("A_init:", A_init, "sigma_init:", sigma_init)
        # print("A:", popt[0], "sigma:", popt[1])
        popt_init_full[i] = [A_init, sigma_init]
        popt_full[i] = popt  # フィッティング結果を格納

    # 各フェーズデータを計算し、対応する配列に格納
        A, sigma = popt_full[i]
        y_values = func_approx(x_axis, A, sigma) + T_room
        temp_apr[i] = y_values  # 横方向に格納
    popt_dict = dict(zip(popt_names, popt_full.T))
    popt_init_dict = dict(zip(popt_init_names, popt_init_full.T))

    return temp_apr, popt_dict, popt_init_dict