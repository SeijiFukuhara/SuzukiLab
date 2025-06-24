import streamlit as st
from flow import FlowAnalyzer  # flow_analyzer.pyにクラスがある場合
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
    use_k_pix = st.checkbox("k_extract_pix_flowを優先する")
    #*移動平均サイズ
    convolve_size_flow = st.number_input('convolve_size_flow', value=21)
    adjust_x_grid = st.number_input("adjust_x_grid", value=0)
    #* 描画の種類を選択
    st.markdown("""### 描画の種類を選択""")
    title_flow = st.checkbox('title_flow')
    is_flow_v_convolve_k_checked = st.checkbox('flow_v_convolve_k')
    is_flow_v_fit_k_checked = st.checkbox('flow_v_fit_k')
    is_flow_v_convolve_fit_k_checked = st.checkbox('flow_v_convolve_fit_k')
    is_flow_v_fit_nobug_k_checked = st.checkbox('flow_v_fit_nobug_k')
    is_flow_v_convolve_fit_nobug_k_checked = st.checkbox('flow_v_convolve_fit_nobug_k')
    #* 描画の種類を選択（divided）
    st.markdown("""### 描画の種類を選択（分割）""")
    title_flow_divided = st.checkbox('title_flow_divided')
    is_flow_v_convolve_k_divided_checked = st.checkbox('flow_v_convolve_k_divided')
    is_flow_v_fit_k_divided_checked = st.checkbox('flow_v_fit_k_divided')
    is_flow_v_convolve_fit_k_divided_checked = st.checkbox('flow_v_convolve_fit_k_divided')
    is_flow_v_fit_nobug_k_divided_checked = st.checkbox('flow_v_fit_nobug_k_divided')
    is_flow_v_convolve_fit_nobug_k_divided_checked = st.checkbox('flow_v_convolve_fit_nobug_k_divided')

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

tab0, tab1, tab2, tab3 = st.tabs(["概要", "1.温度分布", "2.流速分布", "3.熱流束"])


if fname_flow is not None:
    # 一時ファイルとして保存（FlowAnalyzerはファイルパスを必要とする）
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
            use_k_pix=use_k_pix
        )

        # 流速データのネスト辞書を取得
        flow_vy_dict_nested = analyzer.flow_vy_nest_dict
        
        if use_k_pix:
            a = analyzer.k_extract_pix_flow
        else:
            a = analyzer.k_extract_microm_flow
        
        st.write({use_k_pix})
        
        #* 高さkでの流速
        fig = plt.figure()
        if title_flow:
            plt.title(f"Flow Velocity y direction at Height {a} μm")
        if is_flow_v_convolve_k_checked:
            plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                    flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_k'])
        if is_flow_v_fit_k_checked:
            plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                    flow_vy_dict_nested['flow_k_dict']['flow_v_fit_k'])
        if is_flow_v_convolve_fit_k_checked:
            plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                    flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_fit_k'])
        if is_flow_v_fit_nobug_k_checked:
            plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                    flow_vy_dict_nested['flow_k_dict']['flow_v_fit_nobug_k'])
        if is_flow_v_convolve_fit_nobug_k_checked:
            plt.plot(flow_vy_dict_nested['flow_k_dict']['x'], 
                    flow_vy_dict_nested['flow_k_dict']['flow_v_convolve_fit_nobug_k'])
        plt.legend()
        st.pyplot(fig)

        #* 高さkでの流速の分割
        fig = plt.figure()
        if title_flow_divided:
            plt.title(f"Flow Velocity y direction at Height {k_extract_microm_flow}μm (Divided)")
        if is_flow_v_convolve_k_divided_checked:
            plt.plot(flow_vy_dict_nested['flow_k_divided_dict']['x'], 
                    flow_vy_dict_nested['flow_k_divided_dict']['flow_v_convolve_k_divided'])
        if is_flow_v_fit_k_divided_checked:
            plt.plot(flow_vy_dict_nested['flow_k_divided_dict']['x'], 
                    flow_vy_dict_nested['flow_k_divided_dict']['flow_v_fit_k_divided'])
        if is_flow_v_convolve_k_divided_checked:
            plt.plot(flow_vy_dict_nested['flow_k_divided_dict']['x'], 
                    flow_vy_dict_nested['flow_k_divided_dict']['flow_v_convolve_k_divided'])
        if is_flow_v_fit_k_divided_checked:
            plt.plot(flow_vy_dict_nested['flow_k_divided_dict']['x'], 
                    flow_vy_dict_nested['flow_k_divided_dict']['flow_v_fit_k_divided'])
        st.pyplot(fig)



    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
