import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import lu_factor, lu_solve
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from tifffile import TiffFile
from pathlib import Path
from scipy.optimize import curve_fit
import datetime
import csv
import pandas as pd
import math

def loadtext(fname):
    fname_load = np.loadtxt(fname, delimiter = ",")
    return fname_load

def refractive(t):#*参考論文より近似して得た，温度(t)と水の屈折率(ref)の関係．屈折率は温度に関する二次方程式で表される．
    ref = - 0.00000113*t*t - 0.00005285*t + 1.33758
    return ref

def solve_T(a, b, c): #*ref=0とした時の上の方程式を解く．a*x**2 + b*x + c = 0 の解
    D = np.sqrt(abs(b**2 - 4*a*c))
    T = (-b + D) / (2 * a)
    return T

@st.cache_data
#* 書き換えバージョン
def y0(x,popt):
    y0 = popt[0]*np.exp(-popt[2]*(x - popt[1])**2) + x*popt[3] + popt[4]
    return y0

def y1(x,popt):
    y1 = popt[0]*np.exp(-popt[2]*(x - popt[1])**2)
    return y1

def y2(x,popt):
    y2 = popt[0]*np.exp(-popt[2]*x**2)
    return y2

def y3(x,popt):
    y3 = -popt[0]*np.exp(-popt[2]*x**2)
    return y3


def approximation_phase(twolist_array,n,width_phase,height_phase):
    popt_full = np.array([]).reshape(0, 5)  # (0, 5) の空配列
    width_phase_half = width_phase//2 # width_phaseが奇数の場合はエラー出ると思う
    array_x = np.tile(np.array(list(range(-width_phase_half,width_phase_half))), (n,1))
    '''
    width_phase_half = 4、n = 3のとき
    array_x = [[-4 -3 -2 -1  0  1  2  3]
    [-4 -3 -2 -1  0  1  2  3]
    [-4 -3 -2 -1  0  1  2  3]]
    <class 'numpy.ndarray'>
    '''
    def fukuhara_fit(x,a,b,c,d,e):
        #TODO 有効数字の桁数を変えることで結果が大きく変わる
        y = a*np.exp(-c*(x-b)**2) + x*d + e
        return y

    popt_full_list = []
    for i in range(n):
        array_y = np.array(twolist_array[i])
        popt, _ = curve_fit(fukuhara_fit, array_x[0], array_y, maxfev=20000, p0=[1.5, 0, 0.0001, 0.0001, 0.00001])
        popt_full_list.append(popt)  # popt をリストに追加

    popt_full = np.array(popt_full_list) # <class 'list'> -> <class 'nd.aray'>
    p0, p1, p2, p3, p4 = np.split(popt_full, 5, axis=1)
    popt_columns = [p0, p1, p2, p3, p4]
    '''
    popt_full_list = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]] のとき、
    popt_columns = [array([[ 1],[ 4],[ 7],[10]]), array([[ 2],[ 5],[ 8],[11]]), array([[ 3],[ 6],[ 9],[12]])]
    popt_full_listの要素を各列ごとにまとめている
    '''
    phase_full = np.zeros((4, width_phase, height_phase))  # 4つのフェーズデータをまとめて作成

    # y0, y1, y2, y3 をリストに格納
    y_functions = [y0, y1, y2, y3]

    # 各フェーズデータを計算し、対応する配列に格納
    for i, y_func in enumerate(y_functions):
        phase_partial = y_func(array_x, popt_columns)
        phase_full[i, :n, :] = phase_partial

    return phase_full[0], phase_full[1], phase_full[2], phase_full[3], popt_full


def approximation_phase_old(twolist_array,n,width_phase,height_phase):
    listc = []
    list_phase = []
    list_phase2 = []
    list_phase3 = []
    list_phase4 = []
    list_popt_two = []
    def fukuhara_fit(x,a,b,c,d,e):
        #TODO 有効数字の桁数を変えることで結果が大きく変わる
        y = a*np.exp(-c*(x-b)**2) + x*d + e
        return y
    for i in range(n):
        list_y = [] #?繰り返しの中でリストをリセット！！
        list_y2 = []
        list_y3 = []
        list_y4 = []
        list_popt = []
        array_x = np.array(list(range(-int(width_phase/2),int(width_phase/2)))) #*1~1024のリスト
        array_y = np.array(twolist_array[i])
        #print(len(array_x), len(array_y))
        popt, pcov = curve_fit(fukuhara_fit ,array_x, array_y, maxfev = 20000, p0 = [1.5,0,0.0001,0.0001,0.00001])
        for num in range(-int(width_phase/2),int(width_phase/2)):
            list_y.append(popt[0]*np.exp(-popt[2]*(num - popt[1])**2) + num*popt[3] + popt[4])
            list_y2.append(popt[0]*np.exp(-popt[2]*(num - popt[1])**2))
            list_y3.append(popt[0]*np.exp(-popt[2]*(num)**2))
            list_y4.append(-popt[0]*np.exp(-popt[2]*(num)**2))
        list_popt = [popt[0], popt[1], popt[2], popt[3], popt[4]]
        list_phase.append(list_y)
        list_phase2.append(list_y2)
        list_phase3.append(list_y3)
        list_phase4.append(list_y4)
        list_popt_two.append(list_popt)
    #print(len(list_y))
    #print(len(img_phase_apr))
    list_zero = []
    list_zero_popt = []
    for i in range(height_phase - n):
        list_zero = [0] * width_phase
        list_phase.append(list_zero)
        list_phase2.append(list_zero)
        list_phase3.append(list_zero)
        list_phase4.append(list_zero)
        list_zero_popt = [0] * len(list_popt_two[0])
        list_popt_two.append(list_zero_popt)
    list_phase_array = np.array(list_phase)
    list_phase2_array = np.array(list_phase2)
    list_phase3_array = np.array(list_phase3)
    list_phase4_array = np.array(list_phase4)
    #print(popt)
    return list_phase_array, list_phase2_array, list_phase3_array, list_phase4_array, list_popt_two

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
    st.pyplot(fig)

@st.cache_data
#*位相から温度を計算する関数
def calc_temp(twolist_array, Nx, Nz, mode, l, d_temp, n_room, lamda):
    img_phase_array_slice = twolist_array[:,l:Nx]
    Nx = len(img_phase_array_slice[0])
    img_phase_array_slice_flip = np.fliplr(img_phase_array_slice)

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
    #TODO 基本有効だが，場合によってはコメントアウト
    T_solution = np.fliplr(T_solution)
    return T_solution, r

def li_x_temp_offset_func(Nx,l,d_temp):
    li_x_temp_offset = []
    for i in range(l,Nx):
        i = i - Nx
        i = -i /d_temp
        li_x_temp_offset.append(i)
    return li_x_temp_offset

def li_x_temp_apr_func(l,width_phase,d_temp):
    li_x_temp_apr = []
    for i in range(int(width_phase/2)-l):
        li_x_temp_apr.append(i /d_temp)
    return li_x_temp_apr

#*温度のカラーマップを表示する関数
def plot_temp(np_array, d_temp):
    fig = plt.figure()
    plt.imshow(np_array, cmap="rainbow")
    plt.axis('off')
    plt.colorbar(label = "Temperature [℃]")
    plt.clim(23,60)
    scalebar = ScaleBar(1/d_temp,'um',length_fraction = 0.3, location = "upper left")
    plt.gca().add_artist(scalebar)
    st.pyplot(fig)

def show_code():
    st.code("""
    if mode == 0:
        mesh = np.array([[40, 70, 100, 130, 160, Nx],
                        [ 1,  2,   4,   6,   8,  10]])
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
        """)