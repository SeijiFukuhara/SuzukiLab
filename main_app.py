import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
#* sidebarのインポート
from sidebars import sidebar_common, sidebar_temp, sidebar_phase,sidebar_flow
from tabs import tab_phase
from tabs import tab_temp
#* analyzerのインポート
from analyzers.class_phase_analyzer import PhaseAnalyzer
from analyzers.class_temp_analyzer import TempAnalyzer
from analyzers.class_flow_analyzer import FlowAnalyzer
from analyzers.class_heatflux_analyzer import HeatFluxAnalyzer

#* グラフのラベルに日本語を書けるようにするフォント設定
matplotlib.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合


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
    x_adjust,
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
    show_title_temp_extract,
    uniform_filter_size,
    min_points_required
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


# # 計算結果を保存するファイル名
# pickle_filename = "cached_result_temp‘.pkl"
# pickle_filename = "cached_result_flow‘.pkl"

# # UI入力
# param1 = st.slider("Parameter 1", 0, 10, 5)
# param2 = st.slider("Parameter 2", 0, 10, 3)

# # 計算条件のキー（ハッシュ代わり）
# current_params = (param1, param2)

# # 計算結果読み込み関数
# def load_cached_result():
#     if os.path.exists(pickle_filename):
#         with open(pickle_filename, "rb") as f:
#             cached_params, result = pickle.load(f)
#         # パラメータが同じならキャッシュを使う
#         if cached_params == current_params:
#             return result
#     return None

# # 計算結果保存関数
# def save_cached_result(params, result):
#     with open(pickle_filename, "wb") as f:
#         pickle.dump((params, result), f)

# # キャッシュ読み込み
# result = load_cached_result()

# # キャッシュが無ければ計算
# if result is None:
#     st.write("計算を実行します...")
#     # 重い計算処理（例）
#     result = param1 ** 2 + param2 ** 3
#     # 計算結果を保存
#     save_cached_result(current_params, result)
# else:
#     st.write("キャッシュから読み込みました。")

# # 結果表示
# st.write("計算結果:", result)
# import streamlit as st
# import pickle
# import os

# # 計算結果を保存するファイル名
# pickle_filename = "cached_result.pkl"

# # UI入力
# param1 = st.slider("Parameter 1", 0, 10, 5)
# param2 = st.slider("Parameter 2", 0, 10, 3)

# # 計算条件のキー（ハッシュ代わり）
# current_params = (param1, param2)

# # 計算結果読み込み関数
# def load_cached_result():
#     if os.path.exists(pickle_filename):
#         with open(pickle_filename, "rb") as f:
#             cached_params, result = pickle.load(f)
#         # パラメータが同じならキャッシュを使う
#         if cached_params == current_params:
#             return result
#     return None

# # 計算結果保存関数
# def save_cached_result(params, result):
#     with open(pickle_filename, "wb") as f:
#         pickle.dump((params, result), f)

# # キャッシュ読み込み
# result = load_cached_result()

# # キャッシュが無ければ計算
# if result is None:
#     st.write("計算を実行します...")
#     # 重い計算処理（例）
#     result = param1 ** 2 + param2 ** 3
#     # 計算結果を保存
#     save_cached_result(current_params, result)
# else:
#     st.write("キャッシュから読み込みました。")

# # 結果表示
# st.write("計算結果:", result)





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
        microm_or_pix=radio_microm_or_pix,
        x_adjust=x_adjust
    )

    # TempAnalyzer の生成
    tempanalyzer = TempAnalyzer(
        phase_full_array_dict=phaseanalyzer.phase_full_array_dict,
        x_axis=phaseanalyzer.x_axis_pix,
        d_micro_to_pix_temp=d_micro_to_pix_temp,
        k_extract_pix_phase_from_top=phaseanalyzer.k_extract_pix_phase_from_top,
        convolve_size_temp=convolve_size_temp,
        Nx=int(Nx),
        Nz=int(Nz),
        l=int(l),
        n_apr_pix=int(n_apr_pix),
        h0=h0,
        lamda=lamda,
        T_room=T_room,
        target=T_room,
        threshold_cutoff=thredhold_cutoff,
        uniform_filter_size=uniform_filter_size,
        min_points_required=min_points_required
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

# if fname_phase is not None and fname_flow is not None:
#     # HeatFluxAnalyzer の生成
#     heatfluxanalyzer = HeatFluxAnalyzer(
#         tempanalyzer.T_dict,
#         tempanalyzer.x_axis_pix_half,
#         phaseanalyzer.k_extract_pix_phase_from_top,
#         T_room,
#         0.05,
#         int(n_apr_pix),
#         T_room
#     )

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


