import streamlit as st
from class_flow import FlowAnalyzer  # flow_analyzer.pyにクラスがある場合
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt

st.title("FlowAnalyzer Streamlit Viewer")

with st.sidebar:
    st.markdown("## flow.csvファイルを入力")
    fname_flow= st.file_uploader("Choose a flow.csv file",
                                accept_multiple_files= False, type = ['csv'])
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
        d_temp = st.number_input('温度分布観察カメラのレート(1.9833)[pix/μm]', value=1.9833) #*温度分布観察カメラの1umあたりのpixel d[pixel/μm]
        d_micro_to_pix_flow = st.number_input('d_micro_to_pix_flow(1.0269)[pix/μm]', value=1.0269) #*流速分布観察カメラの1umあたりのpixel d[pixel/μm]
        num_zeros = 0

tab01, tab02, tab03 = st.tabs(["1.温度分布", "2.流速分布", "3.熱流束"])

# with tab01:


with tab02:
    if fname_flow is not None:
        temp_path = "temp_uploaded.csv"
        with open(temp_path, "wb") as f:
            f.write(fname_flow.read())

        try:
            # クラスのインスタンスを作成
            analyzer = FlowAnalyzer(
                csv_file=temp_path,
                d_micro_to_pix_flow=d_micro_to_pix_flow,
                k_extract_microm_flow=k_extract_microm_flow,
                adjust_x_grid=adjust_x_grid,
                convolve_size_flow=convolve_size_flow,
                debug_k_pix=debug_k_pix,
                microm_or_pix=radio_microm_or_pix
            )

            # 流速データのネスト辞書を取得
            flow_vy_dict_nested = analyzer.flow_vy_nest_dict
            if radio_microm_or_pix == "k_extract_microm_flow[μm]":
                a = str(analyzer.k_extract_microm_flow) +" μm"
            elif radio_microm_or_pix == "debug_k_pix[pix]":
                a = str(analyzer.k_extract_pix_flow) + " pix"


        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

    else:
        st.warning("flow.csvファイルをアップロードしてください。")

    tab20, tab21 = st.tabs(["グラフ", "csv"])

    with tab20:
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

    with tab21:
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