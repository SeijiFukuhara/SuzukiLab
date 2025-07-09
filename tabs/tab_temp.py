import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

def render_tab_temp(tempanalyzer, phaseanalyzer, T_room, h0, show_phase_temp_dict, show_guide_temp_dict, show_title_temp_extract):
#! 温度分布のタブ
    tab20, tab21, tab22 = st.tabs(["温度分布の図","高さ一定の温度", "温度分布のcsv"])

    #* 温度分布の図
    with tab20:

        i = 0
        for key, fig in tempanalyzer.temp_full_fig_dict.items():
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

                if show_guide_temp_dict['extract_height (k_extract_microm or deug_k_pix)']:
                    ax.axhline(y=phaseanalyzer.k_extract_pix_phase_from_top, color='white', linewidth=1)
                if show_guide_temp_dict['n_apr_pix']:
                    ax.axhline(y=phaseanalyzer.n_apr_pix, color='brown', linewidth=1)
                    ax.axhline(y=phaseanalyzer.height_phase - h0 - 1, color='black', linewidth=1)
                if show_guide_temp_dict['subtract_surface']:
                    ax.axhline(y=phaseanalyzer.height_phase - h0 - 1, color='black', linewidth=1)

                st.pyplot(fig)

            i += 1




        # # 呼び出し側
        # for key, fig_T in tempanalyzer.temp_full_fig_dict.items():
        #     st.write(f"### {key}")
        #     fig, ax = fig_T
        #     ax.axhline(y=phaseanalyzer.k_extract_pix_phase_from_top, color='white', linewidth=1)
        #     ax.axhline(y=phaseanalyzer.height_phase - h0 -1, color='black', linewidth=1)
        #     st.pyplot(fig)

    #* 温度分布の高さ一定グラフ
    with tab21:
        st.markdown("""### 高さ一定温度グラフ""")
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
        if show_title_temp_extract:
            plt.title(f'from substrate {phaseanalyzer.k_extract_microm_phase_from_substrate} [μm]')
        for key, array in tempanalyzer.temp_full_arrey_dict.items():
            if key in show_phase_temp_dict and not show_phase_temp_dict[key]:
                continue
            column = array[phaseanalyzer.k_extract_pix_phase_from_top]
            ax.plot(tempanalyzer.x_axis_pix_half, column, label=key)
        ax.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')  # T=24.5の水平線を引く
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"""### cutoff後の温度分布""")
        fig, ax = plt.subplots()
        if show_title_temp_extract:
            plt.title(f'from substrate {phaseanalyzer.k_extract_microm_phase_from_substrate} [μm]')
        for key, array in tempanalyzer.temp_full_array_cutoff_dict.items():
            if key in show_phase_temp_dict and not show_phase_temp_dict[key]:
                continue
            column = array[phaseanalyzer.k_extract_pix_phase_from_top]
            ax.plot(tempanalyzer.x_axis_pix_cutoff_dict[key][phaseanalyzer.k_extract_pix_phase_from_top],column, label=key)
        plt.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')  # T=24.5の水平線を引く
        plt.legend()
        st.pyplot(fig)
        fig = plt.figure()

        # common_keys = set(tempanalyzer.x_axis_pix_cutoff_dict.keys()) & set(tempanalyzer.temp_full_array_cutoff_dict.keys()) & set(tempanalyzer.temp_full_array_cutoff_apr_dict.keys())
        # a = phaseanalyzer.k_extract_pix_phase_from_top
        # # 各キーに対して処理
        # for key in common_keys:
        #     array1 = tempanalyzer.x_axis_pix_cutoff_dict[key][a]  # 横軸
        #     array2 = tempanalyzer.temp_full_array_cutoff_dict[key][a]  # 縦軸1
        #     array3 = tempanalyzer.temp_full_array_cutoff_apr_dict[key][a]  # 縦軸2
        #     st.markdown(f"""### {key}""")
        #     # プロット
        #     fig, ax = plt.subplots()
        #     ax.plot(array1, array2, label=f"{key} cutoff")
        #     ax.plot(array1, array3, label=f"{key} cutoff apr")
        #     ax.axhline(y=T_room, color='k', linestyle='--',label = 'T = 24.5')
        #     ax.set_xlabel('dict1[{}][{}]'.format(key, a))
        #     ax.set_ylabel('Values')
        #     ax.set_title('Key: {}'.format(key))
        #     ax.legend()
        #     st.pyplot(fig)
        #     plt.close(fig)  # メモリ解放
        
    #* 温度分布のcsv
    with tab22:
        for key, array in tempanalyzer.temp_full_arrey_dict.items():
            st.markdown(f"""### {key}""")
            st.dataframe(pd.DataFrame(array))
        st.markdown("""### cutoff前の温度分布""")
        for key, array in tempanalyzer.temp_full_arrey_dict.items():
            with st.expander(f"{key} cutoff前"):
                st.write(array)
        st.divider()
        # st.markdown("""### cutoff後の温度分布""")
        # for key, array in heatfluxanalyzer.temp_full_array_cutoff_dict.items():
        #     with st.expander(f"{key} cutoff後"):
        #         st.write(array)