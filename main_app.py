import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sidebars import sidebar_common, sidebar_temp, sidebar_phase,sidebar_flow
from tabs import tab_phase
from tabs import tab_temp
# from analyzers_create import (create_phase_analyzer,
#                             create_temp_analyzer,
#                             create_flow_analyzer,
#                             create_heatflux_analyzer)
from analyzers.class_phase_analyzer import PhaseAnalyzer
from analyzers.class_temp_analyzer import TempAnalyzer
from analyzers.class_flow_analyzer import FlowAnalyzer
from analyzers.class_heatflux_analyzer import HeatFluxAnalyzer

#! サイドバー
#* 共通のサイドバー
(
    fname_phase,
    fname_flow,
    k_extract_microm,
    debug_k_pix,
    radio_microm_or_pix,
    T_room,
    lamda,
    d_micro_to_pix_temp,
    d_micro_to_pix_flow
    )= sidebar_common.render_sidebar_common()

#* 位相サイドバー
(
    #* 設定
    Nx,
    Nz,
    n_apr_pix,
    h0,
    l,
    #* offsetの範囲
    z1,
    z2,
    x1,
    x2,
    #* 位相近似の設定
    # gaussian_additive_term,
    convolve_size_temp,
    #* 位相分布のカラーマップ設定
    show_phase_temp_dict,
    show_guide_phase_dict,
    #* 高さ一定の位相
    show_title_phase_dict
    )= sidebar_phase.render_sidebar_phase()

#* 温度サイドバー
(
    meshmode_offset_convolve,
    meshmode_apr,
    temp_offset_convolve,
    temp_apr,
    data_temp_offset_k_extract,
    data_temp_apr_k_extract,
    thredhold_cutoff,
    # show_fig_temp_dict,
    show_guide_temp_dict,
    show_title_temp_extract
    )= sidebar_temp.render_sidebar_temp()

#* 流速サイドバー
# (
#     convolve_size_flow,
#     adjust_x_grid,
#     title_flow,
#     is_flow_vy_k_checked,
#     is_flow_vy_convolve_k_checked,
#     is_flow_vy_fit_k_checked,
#     is_flow_vy_convolve_fit_k_checked,
#     is_flow_vy_fit_nobug_k_checked,
#     is_flow_vy_convolve_fit_nobug_k_checked,
#     title_flow_divided,
#     is_flow_vy_k_divided_checked,
#     is_flow_vy_convolve_k_divided_checked,
#     is_flow_vy_fit_k_divided_checked,
#     is_flow_vy_convolve_fit_k_divided_checked,
#     is_flow_vy_fit_nobug_k_divided_checked,
#     is_flow_vy_convolve_fit_nobug_k_divided_checked,
#     r_min_flow,
#     r_max_flow
#     ) = sidebar_flow.render_sidebar_flow()

#! 解析器のインスタンス生成
if fname_phase is not None:
    temp_phase_path = "temp_uploaded.csv"
    with open(temp_phase_path, "wb") as f:
        f.write(fname_phase.read())

    # PhaseAnalyzer の生成
    phaseanalyzer = PhaseAnalyzer(
        csv_file=temp_phase_path,
        d_micro_to_pix_temp=d_micro_to_pix_temp,
        k_extract_microm=k_extract_microm,
        convolve_size_temp=convolve_size_temp,
        Nx=int(Nx),
        l=int(l),
        h0=int(h0),
        n_apr_pix=int(n_apr_pix),
        z1=z1,
        z2=z2,
        x1=x1,
        x2=x2,
        gaussian_additive_term='linear',
        debug_k_pix=debug_k_pix,
        microm_or_pix=radio_microm_or_pix
    )

    # TempAnalyzer の生成
    tempanalyzer = TempAnalyzer(
        phase_full_arrey_dict=phaseanalyzer.phase_full_array_dict,
        x_axis=phaseanalyzer.x_axis_pix,
        d_micro_to_pix_temp=d_micro_to_pix_temp,
        k_extract_pix_phase_from_top=phaseanalyzer.k_extract_pix_phase_from_top,
        convolve_size_temp=convolve_size_temp,
        Nx=int(Nx),
        Nz=int(Nz),
        l=int(l),
        n_apr_pix=int(n_apr_pix),
        lamda=lamda,
        T_room=T_room,
        target=T_room,
        threshold_cutoff=thredhold_cutoff
    )

if fname_flow is not None:
    temp_flow_path = "temp_uploaded.csv"
    with open(temp_flow_path, "wb") as f:
        f.write(fname_flow.read())

    # # FlowAnalyzer の生成
    # flowanalyzer = FlowAnalyzer(
    #     csv_file=temp_flow_path,
    #     d_micro_to_pix_flow=d_micro_to_pix_flow,
    #     k_extract_microm_flow=k_extract_microm,
    #     adjust_x_grid=adjust_x_grid,
    #     convolve_size_flow=convolve_size_flow,
    #     debug_k_pix=debug_k_pix,
    #     microm_or_pix=radio_microm_or_pix
    # )

if fname_phase is not None and fname_flow is not None:
    # HeatFluxAnalyzer の生成
    heatfluxanalyzer = HeatFluxAnalyzer(
        tempanalyzer.T_dict,
        tempanalyzer.x_axis_pix_half,
        phaseanalyzer.k_extract_pix_phase_from_top,
        T_room,
        0.05,
        int(n_apr_pix),
        T_room
    )

#! タブのレンダリング
tab1, tab2, tab3, tab4 = st.tabs(["位相", "温度", "流速", "熱流束"])

with tab1:
    if fname_phase is not None:
        tab_phase.render_tab_phase(
            phaseanalyzer=phaseanalyzer,
            x1=x1,
            x2=x2,
            z1=z1,
            z2=z2,
            h0=h0,
            show_phase_temp_dict=show_phase_temp_dict,
            show_guide_phase_dict=show_guide_phase_dict,
            show_title_phase_dict=show_title_phase_dict
            )

with tab2:
    if fname_phase is not None:
        tab_temp.render_tab_temp(tempanalyzer=tempanalyzer,
                                phaseanalyzer=phaseanalyzer,
                                T_room=T_room,
                                h0=h0,
                                show_phase_temp_dict=show_phase_temp_dict,
                                show_guide_temp_dict=show_guide_temp_dict,
                                show_title_temp_extract=show_title_temp_extract
                                )


    # st.write(tempanalyzer.temp_full_array_cutoff_apr_dict)
    # st.write(tempanalyzer.x_axis_pix_cutoff_dict)
    # st.write(tempanalyzer.x_axis_pix_cutoff_dict)
    # for key, array_2d in tempanalyzer.temp_full_array_cutoff_dict.items():
    #     st.write(f"### {key} の温度分布")
    #     for i, row in enumerate(array_2d):
    #         if len(row) == 0:
    #             st.write(f"{i} 行目は空です。")

    # for key, array_2d in tempanalyzer.temp_full_array_uniform_cutoff_dict.items():
    #     st.write(f"### {key} の温度分布")
    #     for i, row in enumerate(array_2d):
    #         if len(row) == 0:
    #             st.write(f"{i} 行目は空です。")

    # 空行情報を格納するリスト
    # empty_rows_info = []

    # for key, array_2d in tempanalyzer.temp_full_array_cutoff_dict.items():
    #     for i, row in enumerate(array_2d):
    #         if len(row) == 0:
    #             empty_rows_info.append({"key": key, "row_index": i})

    # # DataFrame に変換
    # df_empty_rows = pd.DataFrame(empty_rows_info)

    # if not df_empty_rows.empty:
    #     st.write("### 空行が存在するキーと行番号（グラフ表示）")
        
    #     # プロット
    #     fig, ax = plt.subplots(figsize=(8, 4))
        
    #     # キーごとにプロット
    #     for key in df_empty_rows['key'].unique():
    #         subset = df_empty_rows[df_empty_rows['key'] == key]
    #         ax.scatter([key]*len(subset), subset['row_index'], label=key, alpha=0.8)
        
    #     ax.set_xlabel("キー")
    #     ax.set_ylabel("空行 index")
    #     ax.set_title("各キーごとの空行分布")
    #     # ax.legend()
    #     plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
        
    #     st.pyplot(fig)
        
    # else:
    #     st.write("空行は存在しません。")

    # tempanalyzer.temp_full_array_cutoff_dict の内容をプロット
    for key, array_2d in tempanalyzer.temp_full_array_cutoff_dict.items():
        # 各行の長さを取得
        row_lengths = [len(row) for row in array_2d]
        
        # プロット
        fig, ax = plt.subplots()
        # ax.plot(row_lengths, marker='o')
        ax.scatter(range(len(row_lengths)), row_lengths, s=5)  # ← scatter に変更
        ax.set_title(f"{key} の各行の要素数")
        ax.set_xlabel("行番号 (index)")
        ax.set_ylabel("要素数 (len(row))")
        
        st.pyplot(fig)


# with tab3:
#     tab_flow.render_tan_flow(None)  # FlowAnalyzer も同様にインスタンス生成して渡す

# with tab4:
#     tab_heatflux.render_heatflux_tab(heatfluxanalyzer)