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
#*読み込んだ流速ファイルから二次元リストを作成
def flow_calculate(path):
    #*データフレームを作成
    df = pd.read_csv(path, encoding="cp932", skiprows = 7, skipfooter = 0, usecols = range(1,7), index_col = None, header = None, engine = 'python')

    df_T = df.T #*作成したデータフレームを転置

    li_T = df_T.to_numpy().tolist() #*df_Tを二次元listに変換（ndarray経由らしい）
    #*リスト中の"-"を0に置換
    for i in range(4,6):
        for j in range(0,len(li_T[0])):
            if li_T[i][j] == "-":
                li_T[i][j] = 0

    li_T_f = np.vectorize(float)(li_T) #*li_Tの要素をすべてfloat型に変換

    return li_T_f

#*FE格子点枠の横幅を取得
def li_width(li):
    width = int(max(li[0]))
    return(width)

#*FE格子点枠の縦幅を取得
def li_height(li):
    height = int(max(li[1]))
    return(height)

#*照射位置のFE格子点座標を取得
def origin_gridpoint(di):
    di2 = di[(float(0), float(0))]
    return [di2[0], di2[1]]

def dic_theta_vr(r0, r1, gridpoint_velocity, x0, y0): #*キーにtheta、値にvrの入った辞書(dic_theta_vr)を作成
    theta = 0
    di = {}
    number = 0
    for key, value in gridpoint_velocity.items():
        x = key[0] - x0
        y = key[1] - y0
        r = math.sqrt(x**2 + y**2)
        if  y > 0 and r0 <= r <= r1:
            theta = math.degrees(math.atan(x/y)) #* 基板より上にある各点においてthetaを算出
            number += 1
            #* 各thetaにおけるvr(r方向の流速)を算出
            vr = value[0]*math.sin(math.radians(theta)) + value[1]*math.cos(math.radians(theta))
            di[theta] = vr
    return di, number

def plot_flow(dic): #*横軸theta、縦軸vrのグラフをプロット
    #* リストxxに対してsize個での移動平均を取る（サイトのコピー）
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

    list_keys = list(dic.keys())
    list_values = list(dic.values())

    #!移動平均とる場合はこの段落を有効に
    #list_vr = valid_convolve(list_vr, 30)

    dic = dict(zip(list_keys,list_values))

    myList = dic.items()
    myList = sorted(myList)
    x, y = zip(*myList)
    return x, y

def plot_flow_directlyabove(): #*横軸照射点直上のピクセル数、縦軸vy(y=const.の線上におけるy方向の流速)のグラフをプロット
    data_plot = li_flow_a_x_y(x0)[y0-1:]
    ylabel = "流速(m/s)"

    #*移動平均をかける関数
    def moving_average(moving):
        global data_plot
        data_plot = data_plot = li_flow_a_x_y(x0)[y0-1:]
        data = np.array(data_plot)
        data_moving_average = np.convolve(data,np.ones(moving)/moving, mode='valid')
        data_plot = data_moving_average
        ylabel = "移動平均(" + str(moving) + ")後の流速(m/s)"
        return ylabel

    #ylabel = moving_average(20) #TODO 移動平均かける場合は有効にする。()は移動平均の定数

    plt.plot(data_plot)
    plt.xlabel('原点からy方向のピクセル数', fontname="MS Gothic")
    plt.ylabel(ylabel,fontname="MS Gothic")

    path_to_file_vr = path_to_file_csv.replace('.csv', 'pix_vr.csv')
    with open(path_to_file_vr, "w", encoding="cp932") as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in data_plot:
            writer.writerow([i])

    #*グラフを書く際の情報をテキストファイルに保存
    r_min_flow_f = r_min_flow * d_micro_to_pix
    r_max_flow_f = r_max_flow * d_micro_to_pix
    print("画像の横幅:" + str(li_width(li_T_f)) + "[pix]", "画像の縦幅:" + str(li_height(li_T_f)) + "[pix]")
    print("バブル原点のx座標:" + str(x0) + "[pix]", "バブル原点のy座標:" + str(y0) + "[pix]")
    print("プロットするデータの個数:" + str(number))
    print("取得する半径の範囲:" + str(r_min_flow) + "[pix]"  + " <= r <= " +  str(r_max_flow) + "[pix]")
    print("取得する半径の範囲:" + str(r_min_flow_f) + "[μm]"  + " <= r <= " +  str(r_max_flow_f) + "[μm]")

    path_to_file_txt = path_to_file_csv.replace('.csv', '.txt')
    f = open(path_to_file_txt, 'w')
    f.write("バブル原点のx座標:" + str(x0) + "[pix]" " " "バブル原点のy座標:" + str(y0) + "[pix]" + "\n" \
    + "プロットするデータの個数:" + str(number) + "\n" \
    + "取得する半径の範囲:" + str(r_min_flow) + "[pix]"  + " <= r <= " +  str(r_max_flow) + "[pix]" + "\n" \
    + "取得する半径の範囲:" + str(r_min_flow_f) + "[μm]"  + " <= r <= " +  str(r_max_flow_f) + "[μm]")
    f.close()

def approximation_flow(dic): #*近似操作をした後のdic_theta_vr（辞書）を返す
    def sincos_fit(x,a,b,c,d):
        y =  -a*np.sin(np.radians(b*x + c))*np.cos(2*np.radians((b*x + c))) + d#*各パラメーターが１桁になるようにオーダーを調整
        return y
    def sincos_fit2(x,a,b,c,d,e,f):
        y =  -a*np.sin(np.radians(b*x + c))*np.cos(2*np.radians((d*x + e))) + f#*各パラメーターが１桁になるようにオーダーを調整
        return y
    array_x = np.array(list(dic.keys()))
    array_y = np.array(list(dic.values()))
    popt, pcov = curve_fit(sincos_fit ,array_x, array_y, p0 = [0.03,1,90,0])
    # popt, pcov = curve_fit(sincos_fit2 ,array_x, array_y, p0 = [0.03,1,90,90,0,0])
    list_y = []
    flow_apr = []
    li_popt = []
    for num in dic.keys():
        list_y.append(-popt[0]*np.sin(np.radians(popt[1]*num + popt[2]))*np.cos(2*np.radians((popt[1]*num + popt[2]))) + popt[3])
    # for num in dic.keys():
        # list_y.append(-popt[0]*np.sin(np.radians(popt[1]*num + popt[2]))*np.cos(2*np.radians((popt[3]*num + popt[4]))) + popt[5])
    # print(popt)
    dic2 = dict(zip(list(dic.keys()), list_y))
    myList = dic2.items()
    myList = sorted(myList)
    x, y = zip(*myList)
    return x, y

def approximation_flow_right(dic): #*片側の近似操作をした後のdic_theta_vr（辞書）を返す
    def exp_fit(x,a,b,c,d,e):
        y = a*np.exp(-c*(x-b)**2) + x*d + e #*各パラメーターが１桁になるようにオーダーを調整
        return y
    array_x = np.array(list(dic.keys()))
    array_y = np.array(list(dic.values()))
    popt, pcov = curve_fit(exp_fit ,array_x, array_y, p0 = [0.03,0,0.1,0.1,-0.005])
    # popt, pcov = curve_fit(sincos_fit2 ,array_x, array_y, p0 = [0.03,1,90,90,0,0])
    list_y = []
    flow_apr = []
    li_popt = []
    for num in dic.keys():
        list_y.append(popt[0]*np.exp(-popt[2]*(num - popt[1])**2) + num*popt[3] + popt[4])
    dic2 = dict(zip(list(dic.keys()), list_y))
    myList = dic2.items()
    myList = sorted(myList)
    x, y = zip(*myList)
    return x, y


def print_and_save(r_min_flow, r_max_flow, d_flow, li_T_f, x0, y0, number, path_to_file_csv): #*使用したデータを表示し、テキストデータで保存
    #*グラフを書く際の情報をテキストファイルに保存
    r_min_flow_f = r_min_flow / d_flow
    r_max_flow_f = r_max_flow / d_flow
    print("画像の横幅:" + str(li_width(li_T_f)) + "[pix]", "画像の縦幅:" + str(li_height(li_T_f)) + "[pix]")
    print("バブル原点のx座標:" + str(x0) + "[pix]", "バブル原点のy座標:" + str(y0) + "[pix]")
    print("プロットするデータの個数:" + str(number))
    print("取得する半径の範囲:" + str(r_min_flow) + "[pix]"  + " <= r <= " +  str(r_max_flow) + "[pix]")
    print("取得する半径の範囲:" + str(r_min_flow_f) + "[μm]"  + " <= r <= " +  str(r_max_flow_f) + "[μm]")

    path_to_file_txt = path_to_file_csv.replace('.csv', '.txt')
    f = open(path_to_file_txt, 'w')
    f.write("バブル原点のx座標:" + str(x0) + "[pix]" " " "バブル原点のy座標:" + str(y0) + "[pix]" + "\n" \
    + "プロットするデータの個数:" + str(number) + "\n" \
    + "取得する半径の範囲:" + str(r_min_flow) + "[pix]"  + " <= r <= " +  str(r_max_flow) + "[pix]" + "\n" \
    + "取得する半径の範囲:" + str(r_min_flow_f) + "[μm]"  + " <= r <= " +  str(r_max_flow_f) + "[μm]")
    f.close()

def grid(li):
    li_gridpoint = list(zip(li[0], li[1]))
    li_coordinates = list(zip(li[2], li[3]))
    li_velocity = list(zip(li[4],li[5]))
    #*キーに照射位置が原点(0,0)のμm単位の座標、バリューにFE格子点座標が入った辞書
    coordinates_gridpoint = {k: v for k, v in zip(li_coordinates, li_gridpoint)}
    #*キーにFE格子点座標、バリューにその点におけるx方向とy方向の流速が入った辞書
    gridpoint_velocity = {k: v for k, v in zip(li_gridpoint, li_velocity)}
    #*レーザー照射位置をFE格子点で表す
    x0 = int(origin_gridpoint(coordinates_gridpoint)[0])
    y0 = int(origin_gridpoint(coordinates_gridpoint)[1])
    return coordinates_gridpoint, gridpoint_velocity, x0, y0

#*vr,thetaの流速グラフを表示する関数
def plot_flow_vr_theta(dic):
    #!横軸theta、縦軸vrのグラフをプロット
    #!すべてのグラフに関する設定
    fig = plt.figure()
    plt.rcParams['xtick.direction'] = 'in' #*x軸目盛を内側に
    plt.rcParams['ytick.direction'] = 'in' #*y軸目盛を内側に
    plt.rcParams["font.size"] = 25 #*すべての文字の大きさを統一（必要に応じて変更）
    x1, y1 = plot_flow(dic) #?元データをそのままプロット(いらない場合はコメントアウト)
    plt.plot(x1, y1, label = '測定データ', lw = 0, marker='o', markersize=5)
    x2, y2 = approximation_flow(dic) #?近似後のデータも重ねてプロット(いらない場合はコメントアウト)
    plt.plot(x2, y2, lw= 4, label = '近似曲線')
    plt.tick_params(width = 3, length = 10)
    plt.xlabel("バブル中心軸からの角度[°]", fontname="MS Gothic")
    plt.ylabel("流速[μm/μs]", fontname="MS Gothic")
    plt.legend(loc=1, prop={'family':'MS Gothic','weight':'light','size':30})
    st.pyplot(fig)
    plt.show() #*グラフ表示

def fit1(data_k,grid_width): #*y = a*np.exp(-c*(x-b)**2) + d*x + e
    def fukuhara_fit(x,a,b,c,d,e):
        #TODO 有効数字の桁数を変えることで結果が大きく変わる
        y = a*np.exp(-c*(x-b)**2) + d*x + e
        return y
    array_x = np.array(list(range(-int(grid_width/2),int(grid_width/2)))) #*1~1024のリスト
    array_y = np.array(data_k)
    popt, pcov = curve_fit(fukuhara_fit ,array_x, array_y, maxfev = 20000, p0 = [0.03,8.5,0.0001,0.01,0.00001])
    li_y = []
    li_y_nobg = []
    li_popt = []
    for num in range(-int(grid_width/2),int(grid_width/2)):
        li_y.append(popt[0]*np.exp(-popt[2]*(num - popt[1])**2) + popt[3]*num + popt[4])
    for num in range(-int(grid_width/2),int(grid_width/2)):
        li_y_nobg.append(popt[0]*np.exp(-popt[2]*(num - popt[1])**2) + popt[4])
    li_popt = [popt[0], popt[1], popt[2], popt[3], popt[4]]
    return li_y, li_y_nobg, li_popt

def rep_fit(data,func,grid_height,grid_width):
    li_flow = []
    li_flow_nobg = []
    li_popt = []
    for i in range(grid_height):
        li_y_k, li_y_k_nobg, li_popt_k = func(data[i],grid_width)
        li_flow.append(li_y_k)
        li_flow_nobg.append(li_y_k_nobg)
        li_popt.append(li_popt_k)
    ar_flow = np.array(li_flow)
    ar_flow_nobg = np.array(li_flow_nobg)
    ar_popt = np.array(li_popt)
    return ar_flow,ar_flow_nobg, ar_popt

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

def rep_convolve(li2, func2, convolve_size_flow):
    li2_convolve = []
    for i in range(len(li2)):
        li2_k_convolve = func2(li2[i], convolve_size_flow)
        li2_convolve.append(li2_k_convolve)
    return li2_convolve

def flow_xy(col, grid_height, grid_width, li):
    li_flow_xy = []
    for i in range(grid_height):
        li_flow_xy_i = []
        for j in range(grid_width):
            li_flow_xy_i.append(li[col][i*grid_width + j])
        li_flow_xy.append(li_flow_xy_i)
    li_flow_xy_reverse = li_flow_xy[::-1]
    return li_flow_xy_reverse

def plot(li2, d_flow):
    fig = plt.figure()
    plt.imshow(li2, cmap="rainbow")
    plt.axis('off')
    # a = plt.colorbar(label = "Corrected Phase [rad]",fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Velocity [m/s]', fontsize=10)
    cbar.ax.tick_params(labelsize=10) #*目盛の数字の大きさ
    #TODO 位相のスケールバーの範囲調節するかも（デフォルトは4.0）
    plt.clim(-0.01,0.04)
    scalebar = ScaleBar(1/d_flow,'um', location = "lower right", length_fraction = 0.2, font_properties={"size": 20}) #*字大きい，位置違う
    #scalebar = ScaleBar(1/d,'um', location = "upper left") #*1 pixel = ? um もともとの設定
    plt.gca().add_artist(scalebar)
    # figname = fname.replace('.csv', '.png') #*保存先のパス．元データのcsvファイルと全く同じファイル名で保存する設定．
    # plt.savefig(figname)
    # plt.show()
    st.pyplot(fig)







# #!熱流束のファイルの関数
# #*y = a*np.exp(-c*(x-b)**2) + d*x + eを用いて関数近似
# def fit1(data_k):
#     def fukuhara_fit(x,a,b,c,d,e):
#         #TODO 有効数字の桁数を変えることで結果が大きく変わる
#         y = a*np.exp(-c*(x-b)**2) + d*x + e
#         return y
#     array_x = np.array(list(range(-int(grid_width/2),int(grid_width/2)))) #*1~1024のリスト
#     array_y = np.array(data_k)
#     popt, pcov = curve_fit(fukuhara_fit ,array_x, array_y, maxfev = 20000, p0 = [0.03,8.5,0.0001,0.01,0.00001])
#     li_y = []
#     li_popt = []
#     for num in range(-int(grid_width/2),int(grid_width/2)):
#         li_y.append(popt[0]*np.exp(-popt[2]*(num - popt[1])**2) + popt[3]*num + popt[4])
#     li_popt = [popt[0], popt[1], popt[2], popt[3]]
#     return li_y, li_popt

# def rep_fit(data,func):
#     li_flow = []
#     li_popt = []
#     for i in range(grid_height):
#         li_y_k, li_popt_k = func(data[i])
#         li_flow.append(li_y_k)
#         li_popt.append(li_popt_k)
#     ar_flow = np.array(li_flow)
#     ar_popt = np.array(li_popt)
#     return ar_flow, ar_popt

# #*移動平均
# def valid_convolve(xx, size):
#     b = np.ones(size)/size
#     xx_mean = np.convolve(xx, b, mode="same")
#     n_conv = math.ceil(size/2)
#     # 補正部分
#     xx_mean[0] *= size/n_conv
#     for i in range(1, n_conv):
#         xx_mean[i] *= size/(i+n_conv)
#         xx_mean[-i] *= size/(i + n_conv - (size % 2))
#     # size%2は奇数偶数での違いに対応するため
#     return xx_mean

# def rep_convolve(li2, func2,convolve_size):
#     li2_convolve = []
#     for i in range(len(li2)):
#         li2_k_convolve = func2(li2[i], convolve_size)
#         li2_convolve.append(li2_k_convolve)
#     return li2_convolve

# def flow_xy(col,li_flow_exp):
#     li_flow_xy = []
#     for i in range(grid_height):
#         li_flow_xy_i = []
#         for j in range(grid_width):
#             li_flow_xy_i.append(li_flow_exp[col][i*grid_width + j])
#         li_flow_xy.append(li_flow_xy_i)
#     li_flow_xy_reverse = li_flow_xy[::-1]
#     return li_flow_xy_reverse

# def plot(li2):
#     plt.figure()
#     plt.imshow(li2, cmap="rainbow")
#     plt.axis('off')
#     a = plt.colorbar(label = "Corrected Phase [rad]")
#     #TODO 位相のスケールバーの範囲調節するかも（デフォルトは4.0）
#     plt.clim(-0.01,0.04)
#     scalebar = ScaleBar(1/d_flow,'um', location = "lower right", length_fraction = 0.2, font_properties={"size": 20}) #*字大きい，位置違う
#     #scalebar = ScaleBar(1/d,'um', location = "upper left") #*1 pixel = ? um もともとの設定
#     plt.gca().add_artist(scalebar)
#     # figname = fname.replace('.csv', '.png') #*保存先のパス．元データのcsvファイルと全く同じファイル名で保存する設定．
#     # plt.savefig(figname)
#     plt.show()

