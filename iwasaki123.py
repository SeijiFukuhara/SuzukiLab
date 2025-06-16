# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:01:10 2023

@author: suzukilab
"""

from PIL import Image
import sys #! コマンドライン
import os #! パスを操作する
import numpy as np
from scipy import signal
from scipy.linalg import lu_factor, lu_solve
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar   #matplotlib_scalebar.scalebar　がエラー出てるけど実行したらできてる，，，
from tifffile import TiffFile
#from scipy.signal import convolve2d #! 移動平均使う場合のみ


#unwrapした位相
file_path = sys.argv[1]  #! コマンドライン引数から取得
img_phase = np.loadtxt(file_path, delimiter=",") #! 20倍の700mA
 #file名の前にRつける
#with TiffFile(R"C:\Users\suzukilab\Desktop\yasuda\shuron\data\20221212\500mA_2000fps_x5_a2\184_0d079s\B_a_000001_phase_r12.tif") as tif:
    #img_phase = tif.asarray()

'''
#移動平均とる
#f = np.full((21,21), 1/(21*21))
#img_phase = signal.convolve2d(img_phase, f, mode='valid')
'''

'''
split_position = 150 #!分割する場所，泡の頂点付近？

upper_part = img_phase[:split_position, :]
lower_part = img_phase[split_position:, :]

kernel = np.ones((10, 40)) / 400
smoothed_lower_part = convolve2d(lower_part, kernel, mode='same', boundary='symm') #! 新たな移動平均

img_phase = np.vstack((upper_part, smoothed_lower_part)) #!　上と下を結合
'''

def refractive(t):
    ref = - 0.00000113*t*t - 0.00005285*t + 1.33758
    return ref

def solve_T(a, b, c):
    D = np.sqrt(abs(b**2 - 4*a*c))
    T = (-b + D) / (2 * a)
    return T

# T_room = 24.5
T_room = 20
lamda = 0.532  #laser wave length [um]
#d = 2.01  #1umあたりのpixel　d[pixel] = 1[um]
d = 8.0  #1umあたりのpixel　d[pixel] = 1[um]
l = 0

n_room = refractive(T_room)

Nx = 540 # 820 #1024 #求めたい温度分布の横の長さ、バブル中心から室温までの長さ pixel単位
Nz = 909 #914 #1000 #求めたい温度分布の縦の長さ
#img_phase = np.fliplr(img_phase) #左ならoff
offset = img_phase[880:900, 920:940] #[z1:z2,x1:x2]の範囲の温度を平均し，その位相を0にoffset、絶対水温の領域を指定
img_phase = offset.mean() - img_phase #位相の逆転を解消
img_phase = np.flipud(np.fliplr(img_phase)) #! ここで180度回転した

#np.savetxt(R"C:\Users\suzukilab\Desktop\yasuda\shuron\data\20221212\500mA_2000fps_x5_a2\184_0d079s\phase_r_crr.txt", img_phase, fmt="%10.5f")
plt.rcParams["font.size"] = 17 #文字の大きさを変更
plt.imshow(img_phase, cmap="rainbow")
plt.axis('off')
plt.colorbar(label = "Corrected Phase [rad]")
plt.clim(0,3.0)
plt.clim(0,4)
scalebar = ScaleBar(1/d,'um', location = "upper left") # 1 pixel = ? um
plt.gca().add_artist(scalebar)
plt.show()

file_dir, file_name = os.path.split(file_path)  # ディレクトリとファイル名を分離
file_base, _ = os.path.splitext(file_name)  # 拡張子を除外
new_file_name2 = f"{file_base}_色付き位相.bmp"  # 新しいファイル名を作成
new_file_name4 = f"{file_base}_色付き位相.csv"  # 新しいファイル名を作成
output_path2 = os.path.join(file_dir, new_file_name2)  # 保存先のフルパスを作成
output_path4 = os.path.join(file_dir, new_file_name4)  # 保存先のフルパスを作成
vmin2 = 0
vmax2 = 4
img_phase = np.clip(img_phase, vmin2, vmax2)
plt.imsave(output_path2, img_phase, cmap='rainbow', format='bmp')
np.savetxt(output_path4, img_phase, fmt='%10.5f', delimiter=",")

plt.figure()#岩崎さんが追加した


img_phase = img_phase[:,l:Nx]
Nx = len(img_phase[0])

img_phase = np.fliplr(img_phase)
T_solution = np.zeros((Nz,Nx))
ref_solution = np.zeros((Nz,Nx))


'''
size_mesh = 10   #pixel
r = np.arange(0, Nx+1, size_mesh)
if (Nx) % size_mesh != 0:
    r = np.append(r, Nx) 

'''
'''
mesh = np.array([[40, 70, 100, 130, 160, Nx],
                 [ 1,  2,   4,   6,   8,  10]])#メッシュサイズ　#! バブル中心からのメッシュサイズ中心の方が密，対物レンズ5倍
'''
mesh = np.array([[130, 160, 190, 220, Nx],
                 [ 1,  2,  4,  6,  10]])#メッシュサイズ　#! バブル中心からのメッシュサイズ中心の方が密，対物レンズ20倍

r = np.array([0])
for i,j in enumerate(mesh[0,:]):
    while r[-1] < j:
        r = np.append(r, r[-1]+(mesh[1,i]))

r = np.delete(r, -1) #!　rの最後の要素がNxを超えているため最後の要素を削除して最後の要素をNxに置き換える
r[-1] = Nx

print(r)

'''[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  42  44  46  48  50  52  54  56  58  60  62  64  66
  68  70  74  78  82  86  90  94  98 102 108 114 120 126 132 140 148 156
 164 174 184 194 204 214 224 234 244 254 264 274 284 294 304 314 324 334
 344 354 364 374 384 394 404 414 424 434 444 454 464 474 484 494 504 520]
 rにこのメッシュ構造を代入した'''

A = np.zeros((r.size-1, r.size-1))  # An = b　#! r[i]からr[i+1]までの光路差を計算，r.sizeはｒの要素の個数
for i in range(r.size-1):
    for j in range(r.size-1):
        if i > j:
            A[i,j] = 0
        else :
            A[i,j] = 2 * np.sqrt((r[j+1]/d)**2 - (r[i]/d)**2) - 2 * np.sqrt((r[j]/d)**2 - (r[i]/d)**2)

for k in range(Nz): #あるz＝一定の平面でTを求めるのを繰り返す
    b = np.zeros(r.size - 1)
    for i in range(r.size-1):
        b[i] = 2 * n_room * np.sqrt((Nx/d)**2 - (r[i]/d)**2) - img_phase[k,r[i]:r[i+1]].mean() * lamda / (2*np.pi)
        #b[i] = 2 * n_room * np.sqrt((Nx/d)**2 - (r[i]/d)**2) - (img_phase[k][r[i]:r[i+1]].mean() - phase_edge) * lamda / (2*np.pi)
    lu_solution = lu_solve(lu_factor(A), b) #屈折率分布　An = bを解く
    
    
    p = 0     #出てきた解を画像の大きさと一緒にするため
    for i in range(r.size-1):
        while p < r[i+1]:
            ref_solution[k,p] = lu_solution[i]
            p += 1
    ref_solution[k, -1] = lu_solution[-1]


    T_solution[k, :] = solve_T(0.00000113, 0.00005285, ref_solution[k, :] - 1.33758)

T_solution[np.isnan(T_solution)] = True  #solve_Tで出た複素解はnanになるので、boolで置換とりあえず
T_solution = np.fliplr(T_solution)  #左ならoff


#np.savetxt(R"/Users/mpepro/Desktop/350mA_20250122_173345端真ん中/480_500_908/10011_phasesss.csv", T_solution, fmt='%10.5f')
plt.imshow(T_solution, cmap="rainbow")
plt.axis('off')
plt.colorbar(label = "Temperature [℃]")
vmin = 20
vmax = 150
plt.clim(vmin,vmax)
#scalebar = ScaleBar(1/d,'um',length_fraction = 0.2, location = "upper left") #! 温度分布のスケールバー
#plt.gca().add_artist(scalebar)
plt.show()

new_file_name3 = f"{file_base}_温度分布.csv"  # 新しいファイル名を作成
output_path3 = os.path.join(file_dir, new_file_name3)  # 保存先のフルパスを作成
np.savetxt(output_path3, T_solution, fmt='%10.5f', delimiter=",")

#! ファイル名を取得して「温度分布」を付加
new_file_name = f"{file_base}_温度分布.bmp"  # 新しいファイル名を作成

output_path = os.path.join(file_dir, new_file_name)  # 保存先のフルパスを作成
T_solution = np.clip(T_solution, vmin, vmax)
plt.imsave(output_path, T_solution, cmap='rainbow', format='bmp')
