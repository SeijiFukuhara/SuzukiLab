import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
#? クラスのインポート
from class_phase import PhaseAnalyzer
from class_flow import FlowAnalyzer

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

    #! 温度分布
    st.markdown('''# :orange[温度分布]''')
    st.markdown("""## 前処理""")
    #TODO 求めたい温度分布の横の長さ，バブル中心から画像横端までの長さ[pix]，0列目からNx列目までを軸対象と仮定する # 1024
    Nx = st.number_input('バブル中心から画像左端の距離：Nx [pix]', value=427)
    #TODO 求めたい温度分布の縦の長さ[pixel] #1000
    Nz = st.number_input('バブル中心から水領域上端の距離；Nz [pix]', value=916)
    n_apr_pix = st.number_input('位相分布を近似する範囲；n_apr_pix [pix]', value = 700)
    h0 = st.number_input('画像下端から基板領域下端までの距離；h0 [pix]', value=129) #*画像下端から基板上面までの距離[pix]
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

    gaussian_additive_term = st.radio("一次関数か定数か：",("linear", "constant"))

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
    #*関数近似を行う位置の基板からの距離[μm]
    k_extract_microm_flow = st.number_input(
        'k_extract_microm_flow[μm]',
        value=100.00,
        step=0.01,
        format="%.2f"
    )
    #*関数近似を行う位置の基板からの距離[pix]
    debug_k_pix = st.number_input('debug_k_pix[pix]', step=1, format="%d")
    radio_microm_or_pix = st.radio("μmかpixか：",("k_extract_microm_flow[μm]", "debug_k_pix[pix]"))

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

tab01, tab02, tab03, tab04 = st.tabs(["1.位相分布","2.温度分布", "3.流速分布", "4.熱流束"])

#! 位相分布のタブ
with tab01:
    if fname_phase is not None:
        temp_phase_path = "temp_uploaded.csv"
        with open(temp_phase_path, "wb") as f:
            f.write(fname_phase.read())
        try:
            # クラスのインスタンスを作成
            phaseanalyzer = PhaseAnalyzer(
                csv_file=temp_phase_path,
                d_micro_to_pix_temp=d_micro_to_pix_temp,
                k_extract_microm_phase=k_extract_microm_phase,
                convolve_size_temp=convolve_size_temp,
                Nx  = int(Nx),
                Nz  = int(Nz),
                n_apr_pix = int(n_apr_pix),
                z1=z1,
                z2=z2,
                x1=x1,
                x2=x2,
                gaussian_additive_term = gaussian_additive_term
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

    else:
        st.warning("phase.csvファイルをアップロードしてください。")

    tab10, tab11, tab12 = st.tabs(["図", "csv", "parameter"])
    #* 位相分布の図
    with tab10:
        if fname_phase is not None:
            st.markdown("""### offset""")
            st.pyplot(phaseanalyzer.fig_offset)
            st.markdown("""### offset_convolve""")
            st.pyplot(phaseanalyzer.fig_offset_convolve)
            fig = plt.figure()
            plt.plot(phaseanalyzer.img_phase_array_offset_convolve[0], color ='blue')
            plt.plot(phaseanalyzer.img_phase_array_offset_convolve[300], color ='red')
            plt.plot(phaseanalyzer.img_phase_array_offset_convolve[500], color ='green')
            st.pyplot(fig)
            if hasattr(phaseanalyzer, 'error_msg') and phaseanalyzer.error_msg:
                st.write("関数近似にエラーが発生:", phaseanalyzer.error_msg)
            else:
                st.markdown("""### offset_convolve_gaussian_plus_linear""")
                fig = phaseanalyzer.fig_offset_convolve_gaussian_plus_linear
                ax = fig.axes[0]
                ax.axhline(y=n_apr_pix, color='white', linewidth=1)
                st.pyplot(fig)

                st.markdown("""### offset_convolve_gaussian""")
                fig = phaseanalyzer.fig_offset_convolve_gaussian
                ax = fig.axes[0]
                ax.axhline(y=n_apr_pix, color='white', linewidth=1)
                st.pyplot(fig)

                st.markdown("""### offset_convolve_gaussian_centered""")
                fig = phaseanalyzer.fig_offset_convolve_gaussian_centered
                ax = fig.axes[0]
                ax.axhline(y=n_apr_pix, color='white', linewidth=1)
                st.pyplot(fig)

            fig = plt.figure()
            plt.plot(phaseanalyzer.x_axis, phaseanalyzer.phase_full[0][500], color ='blue')
            st.pyplot(fig)

    #* 位相分布のcsv
    with tab11:
        if fname_phase is not None:
            st.markdown("""### offset""")
            st.dataframe(pd.DataFrame(phaseanalyzer.img_phase_array_offset))
            st.markdown("""### offset_convolve""")
            st.dataframe(pd.DataFrame(phaseanalyzer.img_phase_array_offset_convolve))
            st.markdown("""### offset_convolve_gaussian_plus_linear""")
            st.dataframe(pd.DataFrame(phaseanalyzer.phase_full[0]))
            st.markdown("""### offset_convolve_gaussian""")
            st.dataframe(pd.DataFrame(phaseanalyzer.phase_full[1]))
            st.markdown("""### offset_convolve_gaussian_centered""")
            st.dataframe(pd.DataFrame(phaseanalyzer.phase_full[2]))
            st.markdown("""### offset_convolve_negative_gaussian_centered""")
            st.dataframe(pd.DataFrame(phaseanalyzer.phase_full[3]))

    #* 位相分布のパラメータ
    with tab12:
        if fname_phase is not None:
            st.markdown("""### A_init""")
            fig = plt.figure()
            plt.plot(phaseanalyzer.popt_init_full[:,0], label = 'A_init', color ='blue')
            plt.plot(phaseanalyzer.popt_full[:,0], label = 'A', color ='red')
            plt.legend()
            st.pyplot(fig)

            st.markdown("""### mu""")
            fig = plt.figure()
            plt.plot(phaseanalyzer.popt_init_full[:,1], label = 'mu_init', color ='blue')
            plt.plot(phaseanalyzer.popt_full[:,1], label = 'mu', color ='red')
            plt.legend()
            st.pyplot(fig)

            st.markdown("""### sigma""")
            fig = plt.figure()
            plt.plot(phaseanalyzer.popt_init_full[:,2], label = 'sigma_init', color ='blue')
            plt.plot(phaseanalyzer.popt_full[:,2], label = 'sigma', color ='red')
            plt.legend()
            st.pyplot(fig)
            if  gaussian_additive_term == "linear":
                st.markdown("""### m""")
                fig = plt.figure()
                plt.plot(phaseanalyzer.popt_init_full[:,3], label = 'm_init', color ='blue')
                plt.plot(phaseanalyzer.popt_full[:,3], label = 'm', color ='red')
                plt.legend()
                st.pyplot(fig)

                st.markdown("""### b""")
                fig = plt.figure()
                plt.plot(phaseanalyzer.popt_init_full[:,4], label = 'b_init', color ='blue')
                plt.plot(phaseanalyzer.popt_full[:,4], label = 'b', color ='red')
                plt.legend()
                st.pyplot(fig)

                st.markdown("""### popt_init""")
                df = pd.DataFrame(phaseanalyzer.popt_init_full,
                                columns=['A_init', 'mu_init', 'sigma_init', 'm_init', 'b_init']) # shape = (4, 1024))
                st.dataframe(df)
                st.markdown("""### popt""")
                df = pd.DataFrame(phaseanalyzer.popt_full,
                                columns = ['A', 'mu', 'sigma', 'm', 'b'])
                st.dataframe(df)

            elif gaussian_additive_term == "constant":
                st.markdown("""### b""")
                fig = plt.figure()
                plt.plot(phaseanalyzer.popt_init_full[:,3], label = 'b_init', color ='blue')
                plt.plot(phaseanalyzer.popt_full[:,3], label = 'b', color ='red')
                plt.legend()
                st.pyplot(fig)

                st.markdown("""### popt_init""")
                df = pd.DataFrame(phaseanalyzer.popt_init_full,
                                columns=['A_init', 'mu_init', 'sigma_init', 'b_init']) # shape = (4, 1024))
                st.dataframe(df)
                st.markdown("""### popt""")
                df = pd.DataFrame(phaseanalyzer.popt_full,
                                columns = ['A', 'mu', 'sigma', 'b'])
                st.dataframe(df)

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
                k_extract_microm_flow=k_extract_microm_flow,
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
        if radio_microm_or_pix == "k_extract_microm_flow[μm]":
            a = str(flowanalyzer.k_extract_microm_flow) +" μm"
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
                plt.title(f"Flow Velocity y direction at Height {k_extract_microm_flow}μm (Divided)")
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