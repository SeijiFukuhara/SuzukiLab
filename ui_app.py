import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
#? クラスのインポート
from class_phase import PhaseAnalyzer
from class_temp import TempAnalyzer
from class_flow import FlowAnalyzer
from class_heatflux import HeatFluxAnalyzer

st.title("熱流束")

with st.sidebar:
    #!ファイル入力
    st.markdown('''# :orange[ファイル入力]''')
    #* 温度分布
    st.markdown("## phase.csvファイルを入力")
    fname_phase = st.file_uploader("Choose a phase.csv file",accept_multiple_files= False, type = ['csv'])
    #* 流速分布
    st.markdown("## flow.csvファイルを入力")
    fname_flow= st.file_uploader("Choose a flow.csv file",accept_multiple_files= False, type = ['csv'])

    st.divider()

    #*関数近似を行う位置の基板からの距離[μm]
    k_extract_microm = st.number_input('基板からの距離 k_extract_microm[μm]',value=100.00,step=0.01,format="%.2f")
    #*関数近似を行う位置の基板からの距離[pix]
    debug_k_pix = st.number_input('画像下端からの距離 debug_k_pix[pix]',min_value=0, step=1, format="%d", value=500)
    radio_microm_or_pix = st.radio("μmかpixか：",("k_extract_microm[μm]", "debug_k_pix[pix]"))

    st.divider()
    #! 温度分布
    st.markdown('''# :orange[温度分布]''')
    st.markdown("""## 設定""")
    #TODO 求めたい温度分布の横の長さ，バブル中心から画像横端までの長さ[pix]，0列目からNx列目までを軸対象と仮定する # 1024
    Nx = st.number_input('バブル中心から画像左端の距離：Nx [pix]', step=1, format="%d", value = 510)
    #TODO 求めたい温度分布の縦の長さ[pixel] #1024
    Nz = st.number_input('温度を計算する範囲；Nz [pix]', step=1, format="%d", value = 1024, max_value=1024)
    n_apr_pix = st.number_input('位相分布を近似する範囲；n_apr_pix [pix]', step=1, format="%d", value = 700)
    h0 = st.number_input('画像下端から基板表面までの距離；h0 [pix]', step=1, format="%d", value = 200) #*画像下端から基板上面までの距離[pix]
    l = st.number_input('端のカット；l [pix]', value=20) #*L[pixel]：phseの画像を端から数えてl(エル)ピクセル目からNxピクセル目までを温度分布にして出力
    #TODO [pix]:近似を行う場合は範囲を指定(<=1024)
    #TODO [z1:z2,x1:x2]の範囲の位相を平均し，その位相を0にoffset，絶対水温の領域を指定．zは縦方向，xは横方向．順番に注意．
    st.markdown("""### [z1&#58;z2,x1&#58;x2]の位相を0にする""")
    z1 = st.number_input('z1[pix]', value=720)
    z2 = st.number_input('z2[pix]', value=730)
    x1 = st.number_input('x1[pix]', value=0)
    x2 = st.number_input('x2[pix]', value=20)
    st.markdown("""## 1. 高さ一定の位相""")
    k_extract_microm_phase = st.number_input('温度，流速を抽出する基板からの高さ；k_extract_microm_phase [μm]', value=100) #*関数近似を行う位置の基板からの距離[μm]
    convolve_size_temp = st.number_input('convolve_size_temp；convolve_size_temp', value=40) #*移動平均サイズ

    gaussian_additive_term = st.radio("一次関数か定数か：",("linear", "constant"), index=0)

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

    #*移動平均サイズ
    convolve_size_flow = st.number_input('convolve_size_flow', value=21)
    adjust_x_grid = st.number_input("adjust_x_grid", value=0)
    #* 描画の種類を選択
    st.markdown("""### 全体""")
    title_flow = st.checkbox('title_flow', value=True)
    is_flow_vy_k_checked = st.checkbox('vy_k')
    is_flow_vy_convolve_k_checked = st.checkbox('vy_convolve_k')
    is_flow_vy_fit_k_checked = st.checkbox('vy_fit_k')
    is_flow_vy_convolve_fit_k_checked = st.checkbox('vy_convolve_fit_k')
    is_flow_vy_fit_nobug_k_checked = st.checkbox('vy_fit_nobug_k')
    is_flow_vy_convolve_fit_nobug_k_checked = st.checkbox('vy_convolve_fit_nobug_k')

    #* 描画の種類を選択（divided）
    st.markdown("""### 分割""")
    title_flow_divided = st.checkbox('title_flow_divided', value=True)
    is_flow_vy_k_divided_checked = st.checkbox('vy_k_divided')
    is_flow_vy_convolve_k_divided_checked = st.checkbox('vy_convolve_k_divided')
    is_flow_vy_fit_k_divided_checked = st.checkbox('vy_fit_k_divided')
    is_flow_vy_convolve_fit_k_divided_checked = st.checkbox('vy_convolve_fit_k_divided')
    is_flow_vy_fit_nobug_k_divided_checked = st.checkbox('vy_fit_nobug_k_divided')
    is_flow_vy_convolve_fit_nobug_k_divided_checked = st.checkbox('vy_convolve_fit_nobug_k_divided')

    st.markdown("""## 3.Vr-Thetaグラフ """)
    r_min_flow = st.number_input('r_min_flow', value=125)
    r_max_flow = st.number_input('r_max_flow', value=126)
    st.divider()
    #!共通
    st.markdown("""# あまり変更しない""")
    with st.sidebar.expander(""):
        T_room = st.number_input('室温[℃]', value=24.5) #*室温
        lamda = st.number_input('観察用レーザーの波長[μm]', value=0.532) #*観察用laser wave length [μm]
        d_micro_to_pix_temp = st.number_input('温度分布観察カメラのレート(1.9833)[pix/μm]', value=1.9833) #*温度分布観察カメラの1umあたりのpixel d[pixel/μm]
        d_micro_to_pix_flow = st.number_input('d_micro_to_pix_flow(1.0269)[pix/μm]', value=1.0269) #*流速分布観察カメラの1umあたりのpixel d[pixel/μm]
        num_zeros = 0

if fname_phase is not None:
    temp_phase_path = "temp_uploaded.csv"
    with open(temp_phase_path, "wb") as f:
        f.write(fname_phase.read())
else:
    st.warning("phase.csvファイルをアップロードしてください。")


tab01, tab02, tab03, tab04 = st.tabs(["1.位相分布","2.温度分布", "3.流速分布", "4.熱流束"])

#! 位相分布のタブ
with tab01:
    if fname_phase is not None:
        # クラスのインスタンスを作成
        phaseanalyzer = PhaseAnalyzer(
            csv_file=temp_phase_path,
            d_micro_to_pix_temp=d_micro_to_pix_temp,
            k_extract_microm=k_extract_microm,
            convolve_size_temp=convolve_size_temp,
            Nx  = int(Nx),
            h0  = int(h0),
            n_apr_pix = int(n_apr_pix),
            z1=z1,
            z2=z2,
            x1=x1,
            x2=x2,
            gaussian_additive_term = gaussian_additive_term,
            debug_k_pix=debug_k_pix,
            microm_or_pix=radio_microm_or_pix
        )

    tab10, tab11, tab12, tab13 = st.tabs(["図","高さ一定の位相", "csv", "parameter"])
    #* 位相分布の図
    with tab10:
        if fname_phase is not None:
            for key, array in phaseanalyzer.phase_full_fig_dict.items():
                st.markdown(f"""### {key}""")
                fig = array
                ax = fig.axes[0]
                ax.axhline(y=phaseanalyzer.k_extract_pix_phase_from_top, color='white', linewidth=1)
                ax.axhline(y=phaseanalyzer.height_phase - h0 -1, color='black', linewidth=1)
                ax.axhline(y=phaseanalyzer.n_apr_pix, color='brown', linewidth=1)
                st.pyplot(fig)
    #* 位相分布の高さ一定グラフ
    with tab11:
            st.markdown(f"""### 高さ一定の位相""")
            st.markdown(f"""### 画面下端から{phaseanalyzer.k_extract_pix_phase_from_bottom}[pix]""")
            st.markdown(f"""### 画面下端から{phaseanalyzer.k_extract_microm_phase_from_bottom}[μm]""")
            st.markdown(f"""### 基板表面から{phaseanalyzer.k_extract_pix_phase_from_substrate}[pix]""")
            st.markdown(f"""### 基板表面から{phaseanalyzer.k_extract_microm_phase_from_substrate}[μm]""")
            fig, ax = plt.subplots()
            for key, array in phaseanalyzer.phase_full_array_dict.items():
                column = array[phaseanalyzer.k_extract_pix_phase_from_top]
                plt.plot(phaseanalyzer.x_axis, column, label=key)
                plt.legend()# キーをラベルに
            st.pyplot(fig)

    #* 位相分布のcsv
    with tab12:
        if fname_phase is not None:
            for key, array in phaseanalyzer.phase_full_array_dict.items():
                st.markdown(f"""### {key}""")
                st.dataframe(pd.DataFrame(array))

    #* 位相分布のパラメータ
    with tab13:
        if fname_phase is not None:
            #? 位相分布のパラメータの初期値と推定値をグラフで表示
            for (popt_init_name, popt_init_value), (popt_name, popt_value) in zip(phaseanalyzer.popt_init_dict.items(), phaseanalyzer.popt_dict.items()):
                st.markdown(f"""### {popt_name}""")
                fig = plt.figure()
                plt.plot(popt_init_value, label = {popt_init_name}, color ='blue')
                plt.plot(popt_value, label = {popt_name}, color ='red')
                plt.legend()
                st.pyplot(fig)
            #? 位相分布のパラメータの初期値と推定値を表で表示
            st.markdown("""### popt_init""")
            df = pd.DataFrame.from_dict(phaseanalyzer.popt_init_dict, orient='index')
            st.dataframe(df)
            st.markdown("""### popt""")
            df = pd.DataFrame.from_dict(phaseanalyzer.popt_dict, orient='index')
            st.dataframe(df)

#! 温度分布のタブ
with tab02:
    tab20, tab21, tab22 = st.tabs(["図", "csv", "高さ一定の温度"])
    if fname_phase is not None:
        try:
            # クラスのインスタンスを作成
            tempanalyzer = TempAnalyzer(
                phase_dict=phaseanalyzer.phase_full_array_dict,
                d_micro_to_pix_temp=d_micro_to_pix_temp,
                k_extract_microm_phase=k_extract_microm_phase,
                convolve_size_temp=convolve_size_temp,
                Nx=int(Nx),
                Nz=int(Nz),
                l=int(l),
                n_apr_pix=int(n_apr_pix),
                lamda=lamda,
                T_room=T_room,
            )
            if hasattr(tempanalyzer, 'error_msg') and tempanalyzer.error_msg:
                st.write("関数近似にエラーが発生:", tempanalyzer.error_msg)
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

    #* 温度分布の図
    with tab20:
        if fname_phase is not None:
            # 呼び出し側
            fig, ax = tempanalyzer.T_fig_dict['offset']
            ax.axhline(y=phaseanalyzer.k_extract_pix_phase_from_top, color='white', linewidth=1)
            ax.axhline(y=phaseanalyzer.height_phase - h0 -1, color='black', linewidth=1)

            # # rの位置に縦線を描画
            # for pos in tempanalyzer.r_dict['offset']:
            #     ax.axvline(x=pos, color='white', linestyle='--', linewidth=0.2)
            st.pyplot(fig)
    #* 温度分布のcsv
    with tab21:
        if fname_phase is not None:
            for key, array in tempanalyzer.T_dict.items():
                st.markdown(f"""### {key}""")
                st.dataframe(pd.DataFrame(array))
    #* 高さ一定の温度
    with tab22:
        if fname_phase is not None:
            fig = plt.figure()
            for key, array in tempanalyzer.T_dict.items():
                plt.plot(array[phaseanalyzer.k_extract_pix_phase_from_top], label = {key})
            plt.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')  # T=24.5の水平線を引く
            plt.legend()
            st.pyplot(fig)

#! 流速分布のタブ
with tab03:
    if fname_flow is not None:
        temp_flow_path = "temp_uploaded.csv"
        with open(temp_flow_path, "wb") as f:
            f.write(fname_flow.read())

        try:
            # クラスのインスタンスを作成
            flowanalyzer = FlowAnalyzer(
                csv_file=temp_flow_path,
                d_micro_to_pix_flow=d_micro_to_pix_temp,
                k_extract_microm_flow=k_extract_microm,
                adjust_x_grid=adjust_x_grid,
                convolve_size_flow=convolve_size_flow,
                debug_k_pix=debug_k_pix,
                microm_or_pix=radio_microm_or_pix
            )
            flow_vy_dict_nested = flowanalyzer.flow_vy_nest_dict

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

    else:
        st.warning("flow.csvファイルをアップロードしてください。")

    if fname_flow is not None:
        if radio_microm_or_pix == "k_extract_microm[μm]":
            a = str(flowanalyzer.k_extract_microm) +" μm"
        elif radio_microm_or_pix == "debug_k_pix[pix]":
            a = str(flowanalyzer.k_extract_pix_flow) + " pix"

    tab30, tab31 = st.tabs(["グラフ", "csv"])

    with tab30:
        if fname_flow is not None:
            #* 高さkでの流速
            st.markdown("""### 基板から高さ一定の流速""")
            fig = plt.figure()
            if title_flow:
                plt.title(f"Flow Velocity y direction at Height {a} ")
            if is_flow_vy_k_checked:
                plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                        flow_vy_dict_nested['flow_k_dict']['flow_v_k'])
            if is_flow_vy_convolve_k_checked:
                plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                        flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_k'])
            if is_flow_vy_fit_k_checked:
                plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                        flow_vy_dict_nested['flow_k_dict']['flow_v_fit_k'])
            if is_flow_vy_convolve_fit_k_checked:
                plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                        flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_fit_k'])
            if is_flow_vy_fit_nobug_k_checked:
                plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                        flow_vy_dict_nested['flow_k_dict']['flow_v_fit_nobug_k'])
            if is_flow_vy_convolve_fit_nobug_k_checked:
                plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                        flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_fit_nobug_k'])
            plt.legend()
            st.pyplot(fig)

            #* 高さkでの流速の分割
            st.markdown("""### 基板から高さ一定の流速（分割）""")
            fig = plt.figure()
            if title_flow_divided:
                plt.title(f"Flow Velocity y direction at Height {k_extract_microm}μm (Divided)")
            if is_flow_vy_convolve_k_divided_checked:
                plt.plot(flow_vy_dict_nested['flow_k_divided_dict']['x'], 
                        flow_vy_dict_nested['flow_k_divided_dict']['flow_v_convolve_k_divided'])
            if is_flow_vy_fit_k_divided_checked:
                plt.plot(flow_vy_dict_nested['flow_k_divided_dict']['x'], 
                        flow_vy_dict_nested['flow_k_divided_dict']['flow_v_fit_k_divided'])
            if is_flow_vy_convolve_k_divided_checked:
                plt.plot(flow_vy_dict_nested['flow_k_divided_dict']['x'], 
                        flow_vy_dict_nested['flow_k_divided_dict']['flow_v_convolve_k_divided'])
            if is_flow_vy_fit_k_divided_checked:
                plt.plot(flow_vy_dict_nested['flow_k_divided_dict']['x'], 
                        flow_vy_dict_nested['flow_k_divided_dict']['flow_v_fit_k_divided'])
            st.pyplot(fig)

    with tab31:
        if fname_flow is not None:
            st.markdown("""### 基板から高さ一定の流速""")
            # 2列に分割
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""###### vy_k  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_dict']['flow_v_k'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)

            with col2:
                st.markdown(f"""###### vy_convolve_k  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_k'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)

            with col3:
                st.markdown(f"""###### vy_fit_k  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_dict']['flow_v_fit_k'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)

            col4, col5, col6 = st.columns(3)

            with col4:
                st.markdown(f"""###### vy_convolve_fit_k  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_fit_k'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)
            with col5:
                st.markdown(f"""###### vy_fit_nobug_k  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_dict']['flow_v_fit_nobug_k'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)
            with col6:
                st.markdown(f"""###### vy_convolve_fit_nobug_k  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_fit_nobug_k'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)

            st.markdown("""### 基板から高さ一定の流速（分割）""")
            # 2列に分割
            col7, col8, col9 = st.columns(3)

            with col7:
                st.markdown(f"""###### vy_k_divided  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_divided_dict']['flow_v_k_divided'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)

            with col8:
                st.markdown(f"""###### vy_convolve_k_divided  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_divided_dict']['flow_v_convolve_k_divided'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)

            with col9:
                st.markdown(f"""###### vy_fit_k_divided  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_divided_dict']['flow_v_fit_k_divided'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)

            col10, col11, col12 = st.columns(3)

            with col10:
                st.markdown(f"""###### vy_convolve_fit_k_divided  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_divided_dict']['flow_v_convolve_fit_k_divided'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)
            with col11:
                st.markdown(f"""###### vy_fit_nobug_k_divided  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_divided_dict']['flow_v_fit_nobug_k_divided'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)
            with col12:
                st.markdown(f"""###### vy_convolve_fit_nobug_k_divided  {a}""")
                df = pd.DataFrame(flow_vy_dict_nested['flow_k_divided_dict']['flow_v_convolve_fit_nobug_k_divided'])
                pd.options.display.float_format = '{:.10f}'.format
                st.dataframe(df)

#! 熱流束のタブ
with tab04:
    # if fname_phase is not None and fname_flow is not None:
    if fname_phase is not None:
        heatfluxanalyzer = HeatFluxAnalyzer(
            tempanalyzer.T_dict,
            phaseanalyzer.k_extract_pix_phase_from_top,
            T_room,
            0.05)
        st.markdown("""### 高さ一定温度グラフ""")
        fig = plt.figure()
        for key, array in tempanalyzer.T_dict.items():
            plt.plot(array[phaseanalyzer.k_extract_pix_phase_from_top], label = {key})
        plt.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')  # T=24.5の水平線を引く
        plt.legend()
        st.pyplot(fig)
        fig = plt.figure()

        for key, array in heatfluxanalyzer.temp_full_array_cutoff_dict.items():
            fig = plt.figure()
            plt.plot(array, label = {key})
            plt.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')  # T=24.5の水平線を引く
            plt.legend()
            st.pyplot(fig)

    else:
        st.warning("phase.csvとflow.csvの両方をアップロードしてください。")