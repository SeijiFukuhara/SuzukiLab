import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

def render_tab_phase(phaseanalyzer, offset_area_slice, h0,show_phase_temp_dict, show_guide_phase_dict, show_title_phase_dict):
    #! 位相分布のタブ
    tab10, tab11, tab12, tab13 = st.tabs(["位相分布の図", "高さ一定の位相", "位相分布のcsv", "位相分布のパラメータ"])
    #* 位相分布の図
    with tab10:

        i = 0
        for key, fig in phaseanalyzer.phase_full_fig_dict.items():
            if key in show_phase_temp_dict and not show_phase_temp_dict[key]:
                continue

            # 奇数番目（0,2,4…）のときに新しい「２列レイアウト」を作る
            if i % 2 == 0:
                cols = st.columns(2)

            # どちらの列に描くかを選択
            col = cols[i % 2]
            with col:
                st.markdown(f"##### {key}")
                ax = fig.axes[0]

                if show_guide_phase_dict['extract_height (k_extract_microm or deug_k_pix)']:
                    ax.axhline(y=phaseanalyzer.k_extract_pix_phase_from_top, color='white', linewidth=1)
                if show_guide_phase_dict['n_apr_pix']:
                    ax.axhline(y=phaseanalyzer.n_apr_pix, color='brown', linewidth=1)
                if show_guide_phase_dict['subtract_surface']:
                    ax.axhline(y=phaseanalyzer.height_phase - h0 - 1, color='black', linewidth=1)
                if show_guide_phase_dict['phase_color_offset_region']:
                    # offset_area_slice から z1,z2,x1,x2 を取り出す
                    z_slice, x_slice = offset_area_slice
                    z1, z2 = z_slice.start, z_slice.stop
                    x1, x2 = x_slice.start, x_slice.stop
                    rect = Rectangle(
                        (x1, z1),
                        x2 - x1,
                        z2 - z1,
                        linewidth=1,
                        edgecolor='r',
                        facecolor="#0e0d0d",
                        alpha=0.3
                    )
                    ax.add_patch(rect)

                st.pyplot(fig)

            i += 1

    #* 位相分布の高さ一定グラフ
    with tab11:
        st.markdown("""### 抽出高さ""")
        shift_pixel = -np.round(phaseanalyzer.popt_dict['mu'][phaseanalyzer.k_extract_pix_phase_from_top]).astype(int)
        shift_microm = round(shift_pixel / phaseanalyzer.d_micro_to_pix_temp, 2)
        data = {
            '単位': ['画面下端から','基板表面から','offset_convolve_centeredで右にずらした量' ],
            'microm': [phaseanalyzer.k_extract_microm_phase_from_bottom, phaseanalyzer.k_extract_microm_phase_from_substrate, shift_microm],
            'pix': [phaseanalyzer.k_extract_pix_phase_from_bottom, phaseanalyzer.k_extract_pix_phase_from_substrate,shift_pixel]
        }
        df = pd.DataFrame(data)
        st.dataframe(df)

        st.markdown("""### 全体グラフ""")
        fig, ax = plt.subplots()
        if show_title_phase_dict:
            plt.title(f'from substrate {phaseanalyzer.k_extract_microm_phase_from_substrate} [μm]')
        for key, array in phaseanalyzer.phase_full_array_dict.items():
                # key が辞書にあって、かつフラグが False ならスキップ
            if key in show_phase_temp_dict and not show_phase_temp_dict[key]:
                continue
            column = array[phaseanalyzer.k_extract_pix_phase_from_top]
            ax.plot(phaseanalyzer.x_axis_pix, column, label=key)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)

        st.markdown(f"""### 温度分布取得に使う範囲のみ""")
        fig, ax = plt.subplots()
        if show_title_phase_dict:
            plt.title(f'from substrate {phaseanalyzer.k_extract_microm_phase_from_substrate} [μm]')
        for key, array in phaseanalyzer.phase_full_array_slice_dict.items():
                # key が辞書にあって、かつフラグが False ならスキップ
            if key in show_phase_temp_dict and not show_phase_temp_dict[key]:
                continue
            column = array[phaseanalyzer.k_extract_pix_phase_from_top]
            ax.plot(phaseanalyzer.x_axis_pix_slice, column, label=key)
        ax.legend()# キーをラベルに
        st.pyplot(fig)

    #* 位相分布のcsv
    with tab12:
        for key, array in phaseanalyzer.phase_full_array_dict.items():
            if key in show_phase_temp_dict and not show_phase_temp_dict[key]:
                continue
            st.markdown(f"""### {key}""")
            st.dataframe(pd.DataFrame(array))

    #* 位相分布のパラメータ
    with tab13:
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
