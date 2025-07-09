import streamlit as st

def render_sidebar_flow():
    with st.sidebar:
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
    return (convolve_size_flow, adjust_x_grid, title_flow, is_flow_vy_k_checked, is_flow_vy_convolve_k_checked, is_flow_vy_fit_k_checked, is_flow_vy_convolve_fit_k_checked, is_flow_vy_fit_nobug_k_checked, is_flow_vy_convolve_fit_nobug_k_checked,
            title_flow_divided, is_flow_vy_k_divided_checked, is_flow_vy_convolve_k_divided_checked, is_flow_vy_fit_k_divided_checked, is_flow_vy_convolve_fit_k_divided_checked, is_flow_vy_fit_nobug_k_divided_checked, is_flow_vy_convolve_fit_nobug_k_divided_checked,
            r_min_flow, r_max_flow)