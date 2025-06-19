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
import csv
import pandas as pd
import math
from scipy import integrate
from datetime import datetime as dt
from PIL import Image
# from datetime import datetime
#!温度計算，流速計算，熱流束計算に必要な関数はそれぞれfunction_temp.py，function_flow.py，function_heat.pyから呼び出し

from function_temp import *
from function_flow import *
from function_temp import *
from function_heatflux import *

#?このサイト全体のタイトル
st.markdown('''# 熱流束''')

#?サイドバー表示
with st.sidebar:
    #!共通データ
    st.markdown('''# :orange[ファイル読み込み]''')
    st.markdown("## phase.csvファイルを入力")
    fname_phase = st.file_uploader("Choose a phase.csv file", accept_multiple_files= False)
    st.markdown("## flow.csvファイルを入力")
    fname_flow= st.file_uploader("Choose a flow.csv file", accept_multiple_files= False)
    st.markdown("## data.xlsxファイルを入力(あれば)")
    data_xlsx= st.file_uploader("Choose a .xlsx file", accept_multiple_files= False)
    st.divider()
    #!温度分布
    st.markdown("""# :orange[温度分布]""")
    st.markdown("""## 前処理""")
    #TODO 求めたい温度分布の横の長さ，バブル中心から画像横端までの長さ[pix]，0列目からNx列目までを軸対象と仮定する # 1024
    Nx = st.number_input('バブル中心から画像左端の距離：Nx [pix]', value=427)
    #TODO 求めたい温度分布の縦の長さ[pixel] #1000
    Nz = st.number_input('バブル中心から水領域上端の距離；Nz [pix]', value=916)
    h0 = st.number_input('画像下端から基板領域下端までの距離；h0 [pix]', value=129) #*画像下端から基板上面までの距離[pix]
    l = st.number_input('端のカット；l [pix]', value=20) #*L[pixel]：phseの画像を端から数えてl(エル)ピクセル目からNxピクセル目までを温度分布にして出力
    #TODO [pix]:近似を行う場合は範囲を指定(<=1024)
    n = Nz #!ここを揃えないと「近似計算後の」温度分布のグラフが描画されない（エラーが出る）
    #TODO [z1:z2,x1:x2]の範囲の位相を平均し，その位相を0にoffset，絶対水温の領域を指定．zは縦方向，xは横方向．順番に注意．
    st.markdown("""### [z1&#58;z2,x1&#58;x2]の位相を0にする""")
    z1 = st.number_input('z1[pix]', value=720)
    z2 = st.number_input('z2[pix]', value=730)
    x1 = st.number_input('x1[pix]', value=0)
    x2 = st.number_input('x2[pix]', value=20)
    st.markdown("""## 1. 高さ一定の位相""")
    k_extract_microm_phase = st.number_input('温度，流速を抽出する基板からの高さ；k_extract_microm_phase [μm]', value=100) #*関数近似を行う位置の基板からの距離[μm]
    convolve_size_temp = st.number_input('convolve_size_temp；convolve_size_temp', value=40) #*移動平均サイズ
    title_phase = st.checkbox('title_phase')
    experiment_plot_phase = st.checkbox('experiment')
    experiment_plot_offset_phase = st.checkbox('experiment_offset')
    experiment_plot_offset_convolve_phase = st.checkbox('experiment_offset_convolve')
    experiment_plot_apr_phase = st.checkbox('experiment_apr')
    experiment_plot_apr_withoutbackground_phase = st.checkbox('experiment_apr_withoutbg')
    experiment_plot_apr_withoutbackground_center_phase = st.checkbox('experiment_apr_withoutbg_center')
    experiment_plot_apr_withoutbackground_center_flip_phase = st.checkbox('approximaton')
    st.markdown("""## 2. 位相カラーマップ""")
    experiment_offset_phase = st.checkbox('1.experiment_offset_phase')
    experiment_offset_convolve_phase = st.checkbox('2.experiment_offset_convolve_phase')
    apr_phase = st.checkbox('3.approximation_phase')
    st.markdown("""## 3. 温度カラーマップ""")
    meshmode_offset_convolve = st.selectbox('meshmode_offset_convolve', (0,1,2,3), index=0)
    meshmode_apr = st.selectbox('meshmode_apr', (0,1,2,3), index=3)
    temp_offset_convolve = st.checkbox('temp_offset_convolve')
    temp_apr = st.checkbox('temp_apr')
    st.markdown("""## 4. 高さ一定の温度""")
    data_temp_offset_k_extract = st.checkbox('data_temp_offset_k_extract')
    data_temp_apr_k_extract = st.checkbox('data_temp_apr_k_extract')
    st.divider()
    #!流速分布
    st.markdown('''# :orange[流速分布]''')
    st.markdown("""## 1.高さ一定の流速""")
    k_extract_microm_flow = st.number_input('流速を抽出する基板からの高さ[μm]', value=100) #*関数近似を行う位置の基板からの距離[μm]
    convolve_size_flow = st.number_input('convolve_size_flow', value=40) #*移動平均サイズ
    title_flow = st.checkbox('title_flow')
    experiment_plot_flow = st.checkbox('experiment_flow')
    experiment_plot_convolve_flow = st.checkbox('experiment_convolve_flow')
    experiment_plot_fit_flow = st.checkbox('experiment_fit_flow')
    experiment_plot_convolve_fit_flow = st.checkbox('experiment_convolve_fit_flow')
    experiment_plot_convolve_fit_nobg_flow = st.checkbox('experiment_convolve_fit_nobg_flow')
    st.markdown("""## 3.Vr-Thetaグラフ """)
    r_min_flow = st.number_input('r_min_flow', value=125)
    r_max_flow = st.number_input('r_max_flow', value=126)
    st.divider()
    #!熱流束
    st.markdown('''# :orange[熱流束]''')
    # k_extract_microm_heatflux = st.number_input('k_extract_microm_heatflux[μm]', value=100)
    # h0 = st.number_input('画像下端から基板上面までの距離[μm]', value=65.1)
    apr_temp_cutoff = st.checkbox('apr_temp_cutoff')
    apr_flow_cutoff = st.checkbox('apr_flow_cutoff')
    k_averagewidth = st.number_input('k_averagewidth [pix]', value=3)
    r0 = st.number_input('レーザースポット半径 [μm]', value=3.6)
    st.divider()
    #!共通
    st.markdown("""# あまり変更しない""")
    with st.sidebar.expander(""):
        T_room = st.number_input('室温[℃]', value=24.5) #*室温
        lamda = st.number_input('観察用レーザーの波長[μm]', value=0.532) #*観察用laser wave length [μm]
        d_temp = st.number_input('温度分布観察カメラのレート(1.9833)[pix/μm]', value=1.9833) #*温度分布観察カメラの1umあたりのpixel d[pixel/μm]
        d_flow = st.number_input('流速分布観察カメラのレート(1.0269)[pix/μm]', value=1.0269) #*流速分布観察カメラの1umあたりのpixel d[pixel/μm]
        num_zeros = 0

tab0, tab1, tab2, tab3 = st.tabs(["概要", "1.温度分布", "2.流速分布", "3.熱流束"])

#!概要
with tab0:
    st.markdown("""
温度分布と流速分布を取得して熱流束を見積もる
""")
    # ファイルがアップロードされた場合の処理
    if data_xlsx is not None:
        # pandasを使用してエクセルファイルを読み込む
        df = pd.read_excel(data_xlsx)
        # データフレームを表示
        st.write("アップロードされたエクセルファイルの内容：")
        st.dataframe(df)
    st.image("figure\streamlit説明.jpg", caption='Streamlitの使い方', use_container_width=True)
#!温度分布
with tab1:
    #st.title('温度分布の算出')
    tab10, tab11, tab12, tab13, tab14, tab15 = st.tabs(["概要", "1.高さ一定の位相", "2.位相分布", "3.温度分布", "4.高さ一定の温度", "5.LOG"])

    #*概要
    with tab10:
        st.markdown("""
            #### 温度分布の概要
            定量位相顕微鏡から得られたphase.csvファイルから
            １. ある高さにおいて位相分布を対称な関数で近似した後のデータプロットと関数
            ２. そのまま変換したphase画像
            ３. 近似関数を用いて再構成したtemp画像とある高さでの横軸左端からの距離，縦軸phaseのグラフを表示，  phase_offset.csv，temp_offset.csvを出力するプログラム
            csvファイルの位相分布はある高さでプロットすると下に凸になる．それを逆転して上に凸にする操作が**offset**
            このプログラムでは**offset**を使わない
            使い方：
            input = extract_phase.pyから得られたphase.csv
            output = フィッティング後の位相分布画像，温度分布画像を表示，フィッティング後の位相分布，温度分布の.csvを保存，横軸左端からの距離，縦軸phaseのグラフ表示
        """)
    #*高さ一定の位相
    with tab11:
        if fname_phase is not None:
            #!phase.csvの読み込み
            img_phase_array = loadtext(fname_phase)
            #?type(img_phase_array) = <class 'numpy.ndarray'>
            #!必要な定数の計算
            n_room = refractive(T_room) #*水の常温=室温として，室温における水の屈折率を算出
            width_phase = len(img_phase_array[0]) #*画像の横幅[pix](1024)
            height_phase = len(img_phase_array) #*画像の縦幅[pix](1024)
            k_extract_pix_phase = height_phase - round(k_extract_microm_phase*d_temp + h0) #*関数近似を行う位置の上端からの距離[pix]0 =< k_extraction =< n (=Nz)
            # #!すべてのグラフに関する設定
            # plt.rcParams['xtick.direction'] = 'in' #*x軸目盛を内側に
            # plt.rcParams['ytick.direction'] = 'in' #*y軸目盛を内側に
            # plt.rcParams["font.size"] = 10 #*すべての文字の大きさを統一（必要に応じて変更）
            # plt.rcParams["font.family"] = "Times New Roman"
            # list_x = list(range(-int(width_phase/2),int(width_phase/2)))
            list_x = list(range(-Nx, width_phase - Nx))
            list_x = list(map(lambda x: x/d_temp, list_x)) #*こっちを有効にしたら画像の中心を原点にする
            img_phase_array_apr, img_phase_array_apr2,img_phase_array_apr3,img_phase_array_apr4, list_popt_two = approximation_phase(img_phase_array,n,width_phase,height_phase) #*近似後のデータを取得
            #print(img_phase_array_apr4) #?二次元リスト
            popt_k_extract = list_popt_two[k_extract_pix_phase][1]
            #print(popt_k_extract)#*高さk_extractにおける頂点のx座標を取得
            fig = plt.figure()
            #? experiment_offset/元データのoffset（移動平均なし）
            img_phase_array_offset = offset(img_phase_array, 1,convolve_size_temp,z1,z2,x1,x2)
            img_phase_array_offset_k = img_phase_array_offset[k_extract_pix_phase]
            #? experiment_offset_convolve/元データのoffset（移動平均あり）
            img_phase_array_offset_convolve = offset(img_phase_array, 0,convolve_size_temp,z1,z2,x1,x2)
            img_phase_array_offset_convolve_k = img_phase_array_offset_convolve[k_extract_pix_phase]
            #? approximation1/近似曲線
            img_phase_array_apr_k = img_phase_array_apr[k_extract_pix_phase]
            #? approximation2/バックグラウンドを外す
            img_phase_array_apr2_k = img_phase_array_apr2[k_extract_pix_phase]
            #? approximation/バックグラウンドを外して原点を中央に移動
            img_phase_array_apr3_k = img_phase_array_apr3[k_extract_pix_phase]
            #? approximation/バックグラウンドを外して原点を中央に移動したものをひっくり返す
            img_phase_array_apr4_k = img_phase_array_apr4[k_extract_pix_phase]
            # plt.style.use('thethis')
            if title_phase:
                plt.title("Distance from Board = " +  str(k_extract_microm_phase) + " [μm]",fontname="Times New Roman ")
            #!高さkにおける横軸横向きのpix，縦軸位相のグラフを，元データと「近似計算後の」データを重ねて表示
            if experiment_plot_phase: #?元データ
                plt.style.use('report')
                plt.plot(list_x, img_phase_array[k_extract_pix_phase].tolist(), label = 'experiment', lw = 0, color = '#1f77b4', marker='o', markersize=3)
            if experiment_plot_offset_phase: #?元データのoffset（移動平均なし）
                plt.plot(list_x, img_phase_array_offset_k, label = 'experiment_offset', lw = 0, color = '#ff7f0e', marker='o', markersize=3)
            if experiment_plot_offset_convolve_phase: #?元データのoffset（移動平均あり）
                plt.plot(list_x, img_phase_array_offset_convolve_k, label = 'experiment_offset/convolve_size = ' + str(convolve_size_temp), lw = 0,color = '#2ca02c', marker='o', markersize=3)
            if experiment_plot_apr_phase: #?近似曲線
                plt.plot(list_x, img_phase_array_apr_k, label = 'experiment_apr', lw = 3, color = '#d62728')
            if experiment_plot_apr_withoutbackground_phase: #?バックグラウンドを外す
                plt.plot(list_x, img_phase_array_apr2_k, label = 'experiment_apr_withoutbg', lw = 3, color = '#9467bd')
            if experiment_plot_apr_withoutbackground_center_phase: #?バックグラウンドを外して原点を中央に移動
                plt.plot(list_x, img_phase_array_apr3_k, label = 'experiment_apr_withoutbg_center', lw = 3, color = '#8c564b')
            if experiment_plot_apr_withoutbackground_center_flip_phase: #?バックグラウンドを外して原点を中央に移動したものをひっくり返す
                plt.plot(list_x, img_phase_array_apr4_k, label = 'approximation', lw = 3, color = '#e377c2')
            #?グラフの詳細設定
            # plt.tick_params(width = 3, length = 6) #*目盛の長さと太さ
            plt.xlabel('Position (\u03bcm)') #*x軸ラベルの指定(日本語の場合はfontnameも指定が必要)
            plt.ylabel('phase (a.u.)') #*y軸ラベルの指定
            # plt.yticks(np.arange(0, 4.5, 0.5)) #*y軸ラベルに関する設定
            plt.legend(loc='upper right',fontsize = 9) #*ラベルを表示させるのに必要
            #plt.title("高さkにおける位相の近似", fontname="MS Gothic")
            plt.tight_layout() #*ラベルの文字などがグラフからはみ出ないように調節
            plt.tick_params(labelsize=10) #*軸の目盛の数字の設定
            st.pyplot(fig)
            #TODOそのうち保存ボタンをつけたい，軸ラベルも自分でつけられるようにしたい
            # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # time_based_filename = fr"C:\Users\seiji\OneDrive - Kyoto University (1)\0.Kyoto.Univ\0.Suzuki_Lab\02.program\CodeRenew\plot_{current_time}.png"
            # save_plot_button(plt.gcf(), filename=time_based_filename)
    #*位相分布
    with tab12:
        if fname_phase is not None:
            st.markdown("""#### 2.1. experiment_offset_phase""")
            data_phase_offset = img_phase_array_offset
            #?type(img_phase_array_offset) = <class 'numpy.ndarray'>
            #!近似計算前の位相分布を表示(experiment_offset)
            if experiment_offset_phase:
                st.markdown("""###### 2.1.1. experiment_offset_phase_data""")
                st.write(pd.DataFrame(data_phase_offset))
                st.markdown("""###### 2.1.2. experiment_offset_phase_plot""")
                plot_phase(data_phase_offset,d_temp)
            else:
                st.markdown("""近似計算前の位相分布を表示(experiment_offset)""")
            #!近似計算前の位相分布を表示(experiment_offset_convolve)
            st.markdown("""#### 2.2.experiment_offset_convolve_phase""")
            data_phase_offset_convolve = img_phase_array_offset_convolve
            if experiment_offset_convolve_phase:
                st.markdown("""###### 2.2.1. experiment_offset_convolve_phase_data""")
                st.write(pd.DataFrame(data_phase_offset_convolve))
                st.markdown("""###### 2.2.2. experiment_offset_convolve_phase_plot""")
                plot_phase(data_phase_offset_convolve,d_temp)
            else:
                st.markdown("""近似計算前の位相分布を表示(experiment_offset_convolve)""")
            #!「近似計算後」の位相分布を表示(approximation)
            st.markdown("""#### 2.3.approximation_phase""")
            data_phase_apr = img_phase_array_apr4
            if apr_phase:
                st.markdown("""###### 2.3.1. approximation_phase_data""")
                st.write(pd.DataFrame(data_phase_apr))
                st.markdown("""###### 2.3.2. approximation_phase_plot""")
                plot_phase(data_phase_apr,d_temp)
            else:
                st.markdown("""「近似計算後」の位相分布を表示(approximation)""")
    #*温度分布
    with tab13:
        st.markdown("""meshmode_offset_convolveとmeshmode_aprで選択するmesh""")
        show_code()
        if fname_phase is not None:
            #!近似計算前の温度分布を表示
            st.markdown("""#### 3.1. temp_offset_convolve""")
            data_temp_offset, mesh_offset = calc_temp(img_phase_array_offset_convolve, Nx, height_phase, meshmode_offset_convolve, l, d_temp, n_room, lamda)
            if temp_offset_convolve:
                #?type(data_temp_offset)classnumpy.ndarray(...)
                string = "convolve_size_temp = " + str(convolve_size_temp)
                st.write(string)
                st.markdown("""##### 3.1.1. temp_offset_convolve_data""")
                st.write(pd.DataFrame(data_temp_offset))
                st.markdown("""##### 3.1.2. temp_offset_convolve_plot""")
                plot_temp(data_temp_offset, d_temp)
            else:
                st.markdown("""近似計算前の温度分布を表示""")
            #!「近似計算後の」温度分布を表示
            st.markdown("""#### 3.2. temp_apr""")
            data_temp_apr, mesh_apr = calc_temp(img_phase_array_apr4, int(width_phase/2), height_phase, meshmode_apr, l, d_temp, n_room, lamda)
            df_data_temp_apr = pd.DataFrame(data_temp_apr)
            if temp_apr:
                #?type(data_temp_apr)classnumpy.ndarray(...)
                st.markdown("""##### 3.2.1. temp_apr_data""")
                st.write(df_data_temp_apr)
                st.markdown("""##### 3.2.2. temp_apr_plot""")
                plot_temp(data_temp_apr, d_temp)
            else:
                st.markdown("""「近似計算後」の温度分布を表示""")
    #*高さ一定の温度分布
    with tab14:
        if fname_phase is not None:
            st.markdown("""#### 1.4.1. data_temp_offset_k_extract""")
            li_x_temp_offset = li_x_temp_offset_func(Nx,l,d_temp)
            if data_temp_offset_k_extract:
                fig = plt.figure()
                # fig,ax = plt.subplots()
                plt.plot(li_x_temp_offset,data_temp_offset[k_extract_pix_phase],label = 'experiment_offset')
                # tex1 = r'$4+2{\rm sin}(2x)$'
                # ax.text(0.2, 1, tex1, fontsize=20, va='bottom', color='C0')
                plt.gca().invert_xaxis()
                plt.axhline(y=T_room, color='k', linestyle='--', label = 'T = 24.5')  # T=24.5の水平線を引く
                plt.legend(fontsize = 9)
                # ax.legend()
                plt.xlabel('Position (\u03bcm)',fontsize = 9) #*x軸ラベルの指定(日本語の場合はfontnameも指定が必要)
                plt.ylabel('temperature (K)',fontsize = 9) #*y軸ラベルの指定
                #plt.title("高さkにおける位相の近似", fontname="MS Gothic")
                plt.tight_layout() #*ラベルの文字などがグラフからはみ出ないように調節
                plt.tick_params(labelsize=9) #*軸の目盛の数字の設定
                st.pyplot(fig)
                st.write(data_temp_offset[k_extract_pix_phase])
            else:
                st.markdown("""offsetした温度分布の高さ一定の分布を表示""")
            st.markdown("""#### 1.4.2. data_temp_apr_k_extract""")
            li_x_temp_apr = li_x_temp_apr_func(l, width_phase,d_temp)
            if data_temp_apr_k_extract:
                fig = plt.figure()
                plt.plot(li_x_temp_apr, data_temp_apr[k_extract_pix_phase][::-1],label = 'apr')
                plt.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')  # T=24.5の水平線を引く
                plt.legend(fontsize = 9)
                plt.xlabel('Position (\u03bcm)', fontsize = 9) #*x軸ラベルの指定(日本語の場合はfontnameも指定が必要)
                plt.ylabel('temperature (K)',fontsize = 9) #*y軸ラベルの指定
                plt.tight_layout() #*ラベルの文字などがグラフからはみ出ないように調節
                plt.tick_params(labelsize=9) #*軸の目盛の数字の設定
                st.pyplot(fig)
                st.write(data_temp_apr[k_extract_pix_phase][::-1])
            else:
                st.markdown("""関数近似した温度分布の高さ一定の分布を表示""")

    with tab15:
        # --- アプリで使う変数（例） ---
        user_name = "Seiji Fukuhara"
        selected_option = "オプションB"
        calculated_value = 42.195

        # --- 表示する変数を辞書などでまとめる ---
        variables = {
            "ユーザー名": user_name,
            "選択肢": selected_option,
            "計算結果": calculated_value
        }

        # --- 画面に表示 ---
        st.title("プログラム中の変数の可視化と保存")

        st.subheader("現在の変数の値")
        for key, value in variables.items():
            st.text(f"{key}: {value}")

        # --- テキストにまとめる ---
        log_content = "\n".join([f"{key}: {value}" for key, value in variables.items()])

        # --- 保存ボタン ---
        st.download_button(
            label="変数の値を保存",
            data=log_content,
            file_name=f"variables_{dt.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )



#!流速分布
with tab2:
    tab20, tab21, tab22, tab23 = st.tabs(["概要", "1.高さ一定の流速", "2.流速カラーマップ", "3.Vr-Thetaグラフ"])

    #*概要
    with tab20:
        st.markdown("""## 概要""")
        st.markdown("""##### ImageJにおける格子点（画像左上が原点）をIJ格子点[pix]，FlowExpertにおける格子点（画像左下が原点）をFE格子点と呼ぶことにする。""")
        st.markdown("""何をしているか？まず得られた流速分布のデータ（FlowExpert）から各高さに対して１ピクセルずつ関数近似を行う．これは各高さにおける分布の中心を求めるためで，分布全体をきれいにフィッティングしたいわけではない．流速分布がきれいに取れなかった場合は左右対称な分布から大きく外れるので，近似した関数は必ずしもきれいにフィットするとは限らない．しかし，関数近似によって得られた中心は，測定した分布の中心と一致すると考えることにする．その後ある高さにおける流速分布を先ほど得た中心位置を基準に片側だけプロットし，これに対して関数近似を行う．今回は分布全体にきれいにフィッティングさせることを目標にする．この場合の関数は必ずしも頂点（原点）での微分係数が０である必要はないのか？？こうして得られた関数をあ る高さにおける流速分布の近似関数とする．""")
        st.markdown("""##### 注意点""")
        st.markdown("""flow_expertで格子点1で解析をしたからといって、必ずしも流速動画.aviのピクセル幅、高さと一致するとは限らない。""")
        if fname_flow is not None:
            #!flow.csvの読み込み
            li_T_f = flow_calculate(fname_flow)
            grid_width = li_width(li_T_f)
            grid_height = li_height(li_T_f)
            coordinates_gridpoint, gridpoint_velocity, x0, y0 = grid(li_T_f)
            dic_theta_vr, number  = make_dic_theta_vr(r_min_flow, r_max_flow,gridpoint_velocity, x0,y0) #*キーにtheta、バリューにvrの辞書を作成
            li_flow_x = flow_xy(4, grid_height, grid_width, li_T_f) #? x方向の流速の２次元リスト(x1)
            li_flow_y = flow_xy(5, grid_height, grid_width, li_T_f) #? y方向の流速の２次元リスト(y1)

            li_flow_x_convolve = rep_convolve(li_flow_x, valid_convolve, convolve_size_flow) #?(x1)に移動平均かけた(x2)
            li_flow_y_convolve = rep_convolve(li_flow_y, valid_convolve, convolve_size_flow) #?(y1)に移動平均かけた(y2)

            ar_flow_y_apr, ar_flow_y_apr_nobg, ar_popt = rep_fit(li_flow_y,fit1,grid_height,grid_width) #?(y1)を関数fit1で近似した(y3)

            # ar_flow_y_convolve_apr, ar_flow_y_convolve_apr_nobg,  ar_popt_convolve = rep_fit(li_flow_y_convolve,fit1,grid_height, grid_width) #?(y2)を関数fit1で近似した(y4)

            k_extract_pix_flow = int(k_extract_microm_flow*d_flow)+y0

            x0_l_flow_pix = int(ar_popt[k_extract_pix_flow][1]+grid_width/2)
            st.write(x0_l_flow_pix)
            li_flow_x_micro = []
            for i in range(1,grid_width+1):
                i -= x0_l_flow_pix
                i /= d_flow
                li_flow_x_micro.append(i)

            li_flow_x_micro_right = []
            for i in range(x0_l_flow_pix+1,grid_width+1):
                i -= x0_l_flow_pix
                i /= d_flow
                li_flow_x_micro_right.append(i)

            Trigonometric_function_x,Trigonometric_function_y = approximation_flow(dict(zip(li_flow_x_micro,li_flow_y_convolve[k_extract_pix_flow])))

            # st.write(ar_flow_y_convolve_apr)
            df_ar_flow_y_convolve_apr = pd.DataFrame(ar_flow_y_convolve_apr)
            #!横軸にバブル直上のpix数、縦軸に流速のグラフをプロット
            #plot_flow_directlyabove()
            #plt.show() #*グラフ表示

            df = pd.DataFrame({
                'd_flow': [str(d_flow),'流速カメラのレート[pix/μm]'],
                'k_extract_microm_phase': [str(k_extract_microm_flow),'流速を抽出する基板からの高さ[μm]'],
                '': [str(k_extract_microm_flow*d_flow),'流速を抽出する基板からの高さ[pix]'],
                'grid_width': [str(grid_width),'FlowExpertの画像の横の長さ[pix]'],
                'grid_height': [str(grid_height),'FlowExpertの画像の縦の長さ[pix]'],
                'x0': [str(x0),'FE格子点におけるレーザー照射位置のx座標[pix]'],
                'y0': [str(y0),'FE格子点におけるレーザー照射位置のy座標[pix]'],
                'convolve_size_flow': [str(convolve_size_flow),'移動平均をかける窓の大きさ'],
            })
            df = df.T
            df.columns = ['値','詳細']
            st.table(df)
            # st.write(df.T)
    #*高さ一定の流速
    with tab21:
        if fname_flow is not None:
            fig, ax = plt.subplots()
            plt.rcParams["font.size"] = 12
            if title_flow:
                plt.title("Distance from Board = " +  str(k_extract_microm_flow) + " [μm]",fontname="Times New Roman ",fontsize=14)
            if experiment_plot_flow:
                plt.plot(li_flow_x_micro,li_flow_y[k_extract_pix_flow], label = 'experiment', lw = 0, color = '#1f77b4', marker='o', markersize=2)
            if experiment_plot_convolve_flow:
                plt.plot(li_flow_x_micro,li_flow_y_convolve[k_extract_pix_flow], label = 'experiment_convolve', lw = 0, color = '#ff7f0e', marker='o', markersize=2)
            if experiment_plot_fit_flow:
                plt.plot(li_flow_x_micro,ar_flow_y_apr[k_extract_pix_flow], label = 'experiment_fit', lw = 2, color = '#2ca02c', marker='o', markersize=0)
            if experiment_plot_convolve_fit_flow:
                plt.plot(li_flow_x_micro,ar_flow_y_convolve_apr[k_extract_pix_flow], label = 'experiment_convolve_fit', lw = 2, color = '#d62728', marker='o', markersize=0)
            if experiment_plot_convolve_fit_nobg_flow:
                plt.plot(li_flow_x_micro,ar_flow_y_convolve_apr_nobg[k_extract_pix_flow], label = 'experiment_convolve_fit_nobg', lw = 2, color = '#9467bd', marker='o', markersize=0)

            # plt.plot(Trigonometric_function_x,Trigonometric_function_y, label = 'sincos', lw = 2, color = '#9467bd', marker='o', markersize=0)

            plt.axhline(y=0, color='k', linestyle='--')  # vy=0の水平線を引く
            plt.legend(loc='upper right',fontsize = 9)
            plt.tick_params(labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            # st.write(f"<span style='font-size:30px'>{"y = a*exp(-c*(x-b)^2) + d*x + e"}</span>", unsafe_allow_html=True)
            df = pd.DataFrame({
                'experiment_fit': [ar_popt[k_extract_pix_flow][0],ar_popt[k_extract_pix_flow][1],ar_popt[k_extract_pix_flow][2],ar_popt[k_extract_pix_flow][3],ar_popt[k_extract_pix_flow][4]],
                'experiment_convolve_fit': [ar_popt_convolve[k_extract_pix_flow][0],ar_popt_convolve[k_extract_pix_flow][1],ar_popt_convolve[k_extract_pix_flow][2],ar_popt_convolve[k_extract_pix_flow][3],ar_popt_convolve[k_extract_pix_flow][4]],
            })
            df = df.T
            df.columns = ['a','b','c','d','e']
            st.table(df)

            apr_flow_right_x, apr_flow_right_y =approximation_flow_right(dict(zip(li_flow_x_micro_right,li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:])))

            fig, ax = plt.subplots()
            plt.rcParams["font.size"] = 12
            plt.plot(li_flow_x_micro_right,li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:], label = 'experiment_convolve_fit_nobg', lw = 0, color = '#1f77b4', marker='o', markersize=2)
            # plt.plot(apr_flow_right_x,apr_flow_right_y, label = 'experiment_convolve_fit_nobg', lw = 2, color = '#ff7f0e', marker='o', markersize=0)
            plt.axhline(y=0, color='k', linestyle='--')  # vy=0の水平線を引く
            plt.legend(loc='upper right',fontsize = 9)
            plt.tick_params(labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)
    #*流速カラーマップ
    with tab22:
        if fname_flow is not None:
            st.markdown("""#### FlowExpertのデータそのまま（z方向）""")
            plot(li_flow_y, d_flow) #?(y1)
            st.markdown("""#### 関数近似""")
            plot(ar_flow_y_apr, d_flow) #?(y2)
            st.markdown("""#### 関数近似に移動平均かけた""")
            l
            plot(ar_flow_y_convolve_apr, d_flow) #?(y3)
            st.markdown("""#### FlowExpertのデータそのまま（x方向）""")
            plot(li_flow_x, d_flow) #?(y4)
    #*Vr-Thetaグラフ
    with tab23:
        if fname_flow is not None:
            #!横軸theta、縦軸vrのグラフをプロット
            plot_flow_vr_theta(dic_theta_vr)
            st.markdown("""近似曲線を描く意味はあんまりないか？""")

# #!熱流束
with tab3:
    tab30, tab31, tab32, tab33 = st.tabs(["データ", "1.範囲切り取り無し", "2.範囲切り取り", "3.計算結果"])
    if fname_phase and fname_flow is not None and k_extract_microm_phase == k_extract_microm_flow:
        with tab30:
            #!温度に関する情報
            len_pix_temp = len(data_temp_apr[k_extract_pix_phase])
            len_microm_temp = len_pix_temp/d_temp
            range_microm_temp = np.count_nonzero(np.round(data_temp_apr[k_extract_pix_phase][::-1], 5) > T_room)/d_temp
            range_temp_pix = int(range_microm_temp*d_temp)
            #!流速に関する情報
            len_pix_flow = len(li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:])
            len_microm_flow = len(li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:])/d_flow
            range_microm_flow = np.count_nonzero(np.round(li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:], 5) > 0)/d_flow
            range_pix_flow = int(range_microm_flow*d_flow)
            #!情報の出力
            df = pd.DataFrame({
                '': [k_extract_microm_phase,'μm','計算対象とする高さ'],
                'd_temp': [str(d_temp),'pix/μm','温度カメラのレート'],
                'len_pix_temp': [str(len_pix_temp),'pix','温度分布の取得範囲'],
                'len_microm_temp': [len_microm_temp,'μm','温度分布の取得範囲'],
                'd_flow': [d_flow,'pix/μm','流速カメラのレート'],
                'len_pix_flow': [len_pix_flow,'pix','流速分布の取得範囲'],
                'len_microm_flow': [len_microm_flow,'μm','流速分布の取得範囲'],
                'range_microm_temp': [range_microm_temp,'μm','水温が室温より高い範囲'],
                'range_temp_pix': [range_temp_pix,'pix','水温が室温より高い範囲'],
                'range_microm_flow': [range_microm_flow,'μm','流速が0.0より大きい範囲'],
                'range_pix_flow': [range_pix_flow,'pix','流速が0.0より大きい範囲'],
            })
            df = df.T
            df.columns = ['値','単位','詳細']
            st.table(df)
            st.markdown("""##### 温度がT_roomより高いのはバブルから""")
            st.markdown(str(range_microm_temp))
            st.markdown("""##### [μm]の距離にある領域です""")
            #!水温が室温以上の領域における温度分布の近似
            x_apr_temp_cutoff, y_apr_temp_cutoff, popt_temp = approximation_temp_cutoff(dict(zip(li_x_temp_apr[:int(range_microm_temp*d_temp)], data_temp_apr[k_extract_pix_phase][::-1][:int(range_microm_temp*d_temp)])), T_room)
            #!水温が室温以上の領域における流速分布の近似
            x_apr_flow_cutoff, y_apr_flow_cutoff, popt_flow = approximation_flow_cutoff(dict(zip(li_flow_x_micro_right[:int(range_microm_temp*d_flow)], li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:][:int(range_microm_temp*d_flow)])))
            #!温度分布の平均をとる
            empty_array = []
            for i in range(k_extract_pix_phase-k_averagewidth,k_extract_pix_phase+k_averagewidth):
                empty_array.append(data_temp_apr[i])
            st.write(empty_array)
            data_temp_apr[k_extract_pix_phase] = np.mean(empty_array, axis=0)
            st.write(data_temp_apr[k_extract_pix_phase])
        with tab31:
            st.markdown("""#### 3.1.1. 温度""")
            fig,ax = plt.subplots()
            plt.plot(li_x_temp_apr, data_temp_apr[k_extract_pix_phase][::-1],label = 'apr', lw = 0, color = '#939393', marker='o', markersize=2)
            plt.plot(li_x_temp_apr[:range_temp_pix], data_temp_apr[k_extract_pix_phase][::-1][:range_temp_pix],label = 'apr', lw = 0, color = '#1f77b4', marker='o', markersize=2)
            if apr_temp_cutoff:
                plt.plot(x_apr_temp_cutoff, y_apr_temp_cutoff, label = 'apr2',lw = 2, color = '#ff7f0e')
            plt.plot(li_x_temp_apr[range_temp_pix], data_temp_apr[k_extract_pix_phase][::-1][range_temp_pix],color = '#ff1493', marker='o', markersize=5)
            y_range_min, y_range_max = ax.get_ylim()
            plt.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')  # T=24.5の水平線を引く
            # plt.axvline(x=range_microm_temp,ymax=(T_room-y_range_min)/(y_range_max-y_range_min) ,color='#ff1493', linestyle='--')
            plt.legend(fontsize = 9)
            plt.xlabel('Position (\u03bcm)', fontsize = 9) #*x軸ラベルの指定(日本語の場合はfontnameも指定が必要)
            plt.ylabel('temperature (K)',fontsize = 9) #*y軸ラベルの指定
            plt.tight_layout() #*ラベルの文字などがグラフからはみ出ないように調節
            plt.tick_params(labelsize=9) #*軸の目盛の数字の設定
            st.pyplot(fig)

            st.markdown("""#### 3.1.2. 流速""")
            y_temp = li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:][int(range_microm_temp/d_flow)]
            fig, ax = plt.subplots()
            plt.rcParams["font.size"] = 12
            # plt.plot(range_microm_temp,y_temp,marker='.',markersize=10)
            plt.text(range_microm_temp-10, y_temp+0.001, '({x}, {y})'.format(x=np.round(range_microm_temp,2), y=np.round(y_temp,7)), fontsize=10)
            plt.plot(li_flow_x_micro_right[int(range_temp_pix/d_temp*d_flow)], li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:][int(range_temp_pix/d_temp*d_flow)],color = '#ff1493', marker='o', markersize=5)
            plt.plot(li_flow_x_micro_right,li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:], label = 'experiment_convolve_fit_nobg', lw = 0, color = '#939393', marker='o', markersize=2)
            plt.plot(li_flow_x_micro_right[:int(range_temp_pix/d_temp*d_flow)],li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:][:int(range_temp_pix/d_temp*d_flow)], label = 'experiment_convolve_fit_nobg', lw = 0, color = '#1f77b4', marker='o', markersize=2)
            if apr_flow_cutoff:
                plt.plot(x_apr_flow_cutoff, y_apr_flow_cutoff, label = 'experiment_convolve_fit_nobg', lw = 2, color = '#ff7f0e', marker='o', markersize=0)
            y_range_min, y_range_max = ax.get_ylim()
            # plt.axvline(x=range_microm_temp,ymax=(y_temp-y_range_min)/(y_range_max-y_range_min)-0.01 ,color = '#ff1493', linestyle='--')
            plt.axhline(y=0, color='k', linestyle='--',label = 'Vr =0.000')  # vy=0の水平線を引く
            plt.legend(loc='upper right',fontsize = 9)
            plt.tick_params(labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        with tab32:
            st.markdown("""#### 3.2.1. 温度""")
            fig,ax = plt.subplots()
            # plt.text(range_microm_temp-10, T_room+1.0, '({x}, {y})'.format(x=np.round(range_microm_temp,2), y=T_room), fontsize=10)
            plt.plot(li_x_temp_apr[:int(range_microm_temp*d_temp)], data_temp_apr[k_extract_pix_phase][::-1][:int(range_microm_temp*d_temp)],label = 'apr',lw = 0, color = '#1f77b4', marker='o', markersize=2)
            plt.plot(x_apr_temp_cutoff, y_apr_temp_cutoff, label = 'apr2',lw = 2, color = '#ff7f0e')
            y_range_min, y_range_max = ax.get_ylim()
            plt.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')  # T=24.5の水平線を引く
            plt.plot(li_x_temp_apr[range_temp_pix], data_temp_apr[k_extract_pix_phase][::-1][range_temp_pix],color = '#ff1493', marker='o', markersize=5)
            # plt.axvline(x=range_microm_temp,ymax=(T_room-y_range_min)/(y_range_max-y_range_min) ,color='#ff1493', linestyle='--')
            plt.legend(fontsize = 9)
            plt.xlabel('Position (\u03bcm)', fontsize = 9) #*x軸ラベルの指定(日本語の場合はfontnameも指定が必要)
            plt.ylabel('temperature (K)',fontsize = 9) #*y軸ラベルの指定
            plt.tight_layout() #*ラベルの文字などがグラフからはみ出ないように調節
            plt.tick_params(labelsize=9) #*軸の目盛の数字の設定
            st.pyplot(fig)

            # st.write(f"<span style='font-size:30px'>{"y = a*exp(-c*(x-b)^2)"}</span>", unsafe_allow_html=True)
            df = pd.DataFrame({
                'popt': [popt_temp[0],popt_temp[1],popt_temp[2]]
            })
            df = df.T
            df.columns = ['a','b','c']
            st.table(df)

            st.markdown("""#### 3.2.2. 流速""")
            y_temp = li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:][int(range_microm_temp/d_flow)]
            fig, ax = plt.subplots()
            plt.rcParams["font.size"] = 12
            # plt.text(range_microm_temp-20, y_temp+0.001, '({x}, {y})'.format(x=np.round(range_microm_temp,2), y=np.round(y_temp,7)), fontsize=10)
            plt.plot(li_flow_x_micro_right[:int(range_microm_temp*d_flow)],li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:][:int(range_microm_temp*d_flow)], label = 'experiment_convolve_fit_nobg', lw = 0, color = '#1f77b4', marker='o', markersize=2)
            plt.plot(x_apr_flow_cutoff, y_apr_flow_cutoff, label = 'experiment_convolve_fit_nobg', lw = 2, color = '#ff7f0e', marker='o', markersize=0)
            y_range_min, y_range_max = ax.get_ylim()
            plt.plot(li_flow_x_micro_right[int(range_temp_pix/d_temp*d_flow)], li_flow_y_convolve[k_extract_pix_flow][x0_l_flow_pix:][int(range_temp_pix/d_temp*d_flow)],color = '#ff1493', marker='o', markersize=5)
            plt.axhline(y=0, color='k', linestyle='--',label = 'Vr =0.000')  # vy=0の水平線を引く
            plt.legend(loc='upper right',fontsize = 9)
            plt.tick_params(labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)

            # st.write(f"<span style='font-size:30px'>{"y = a*exp(-c*(x-b)^2) + d*x + e"}</span>", unsafe_allow_html=True)
            df = pd.DataFrame({
                'popt': [popt_flow[0],popt_flow[1],popt_flow[2],popt_flow[3],popt_flow[4]]
            })
            df = df.T
            df.columns = ['a','b','c','d','e']
            st.table(df)

        with tab33:
            C_p = 4.186 #?[J/gK] 水の低圧比熱容量
            rho = 1.0 * 10**3 #?[kg/m^3] 水の密度
            pi = np.pi
            def f(r):
                #*温度の関数
                [a, b, c] = popt_temp
                y_temp = a*np.exp(-c*(r-b)**2)
                #*流速の関数
                [a, b, c, d, e] = popt_flow
                y_flow = a*np.exp(-c*(r-b)**2) + d*r + e
                y = 2 * pi * C_p * rho * (y_flow * y_temp) * r * 10**(-9)
                #*流れの関数*温度の関数*水の比熱*面積分(rはμm)
                return y

            st.markdown("""#### 3.3.1. 計算式""")


            # 画像ファイルを読み込む
            image = Image.open('./images/formula.png')
            # Streamlitで画像を表示する
            st.image(image, use_column_width=True)

            arr = np.arange(100)
            empty_array = np.empty((0,))
            fig = plt.figure()
            for i in arr:
                empty_array = np.append(empty_array, f(i))
            plt.plot(arr,empty_array)
            # st.pyplot(fig)

            ans,err = integrate.quad(f,0, range_microm_temp)
            Q = np.round(ans * 10**3,4)
            q = Q*10**(-3)/(pi*(r0*10**(-6))**2)
            #*メモ
            #*y_temp[K]
            #*y_flow[μm/μs]

            df = pd.DataFrame({
                'k_extract_microm_phase': [k_extract_microm_phase,'μm','熱計算を行う位置の基板からの距離'],
                'C_p': [C_p,'J/gK','水の低圧比熱容量'],
                'rho': [rho,'kg/m^3','水の密度'],
                'r_max': [np.round(range_microm_temp,4),'μm','水の密度'],
                'Q': [Q,'mW','熱量'],
                'r0': [r0,'μm','レーザースポット半径'],
                'q': ['{:.4e}'.format(q),'W/m^2','レーザー照射位置における熱流束'],
            })
            df = df.T
            df.columns = ['値','単位','詳細']
            # st.table(df)

            now = dt.now()
            # 指定されたフォーマットで日付と時刻を文字列に変換します
            formatted_datetime = now.strftime("%Y%m%d%H%M%S")

            st.markdown( """ #### 3.3.2 条件と計算結果""")
            # データフレームを表示します
            st.write(df)

            # エクセルファイルをダウンロードします
            df_with_index = df.reset_index()
            # エクセルファイルにデータフレームを保存します
            excel_file = df_with_index.to_excel("sample_excel_file.xlsx", index=False)
            st.download_button(
                label="Download",
                data=open("sample_excel_file.xlsx", "rb"),
                file_name="heatflux_" + str(formatted_datetime) + ".xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
            st.markdown( """#### k_extract_microm_phase と k_extract_microm_flowを一致させてください""")