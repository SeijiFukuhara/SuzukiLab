import streamlit as st

def render_sidebar_temp():
    with st.sidebar:
        #! 温度分布
        st.markdown('''# :orange[温度分布]''')
        
        st.markdown("""## :green[温度分布の図]""")
        show_guide_dict = {
            'extract_height (k_extract_microm or deug_k_pix)' : st.checkbox('show_phase_color_extract_height (k_extract_microm or deug_k_pix)', key = 'guide_extract_height_temp', value=False),
            'n_apr_pix' : st.checkbox('show_phase_color_n_apr_pix', key = 'guide_n_apr_pix_temp', value=False),
            'subtract_surface' : st.checkbox('show_phase_color_subtract_offset (h0)', key = 'guide_subtract_surface_temp', value=False)
        }

        st.markdown("""## :green[高さ一定の温度]""")
        show_title_temp_extract = st.checkbox('title_temp_extract')
        st.markdown("""## 3. 温度カラーマップ""")
        meshmode_offset_convolve = st.selectbox('meshmode_offset_convolve', (0,1,2,3), index=0)
        meshmode_apr = st.selectbox('meshmode_apr', (0,1,2,3), index=3)
        temp_offset_convolve = st.checkbox('temp_offset_convolve')
        temp_apr = st.checkbox('temp_apr')
        data_temp_offset_k_extract = st.checkbox('data_temp_offset_k_extract')
        data_temp_apr_k_extract = st.checkbox('data_temp_apr_k_extract')
        st.divider()
        thredhold_cutoff = st.number_input('thredhold_cutoff',value=0.1,step=0.01,format="%.2f")

    return (
        meshmode_offset_convolve,
        meshmode_apr,
        temp_offset_convolve,
        temp_apr,
        data_temp_offset_k_extract,
        data_temp_apr_k_extract,
        thredhold_cutoff,
        # show_fig_temp_dict,
        show_guide_dict,
        show_title_temp_extract
    )