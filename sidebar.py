import streamlit as st
import os
import glob

#* streamlitを更新すると、通常は変数はデフォルト値に戻されてしまう。
#* しかし、この関数ではsession_stateを使用することで、変数の値を保持することができる。
#* （UIを更新しても前回の入力値がそのまま入力される）
def _number_input_with_session_state(label, key, default_value, step=1, format="%d"):
    """
    Streamlit number_input において、session_state を初期化して使用する関数。

    Parameters
    ----------
    label : str
        number_input のラベル
    key : str
        session_state に保存するキー
    default_value : int or float
        初期値
    step : int or float, optional
        ステップ幅
    format : str, optional
        表示フォーマット

    Returns
    -------
    value : int or float
        入力された値
    """

    if key not in st.session_state:
        st.session_state[key] = default_value

    value = st.number_input(
        label,
        step=step,
        format=format,
        value=st.session_state[key],
        key=key
    )

    return value

def render_sidebar_common():
    with st.sidebar:
        #!ファイル入力
        st.markdown('''# :orange[ファイル入力]''')
        #* 温度分布
        st.markdown("## phase.csvファイルを入力")
        fname_phase = st.file_uploader("Choose a phase.csv file",accept_multiple_files= False, type = ['csv'])
        #* 流速分布
        st.markdown("## flow.csvファイルを入力")
        fname_flow= st.file_uploader("Choose a flow.csv file",accept_multiple_files= False, type = ['csv'])

        # 削除対象のディレクトリ
        target_folder = "cache_folder"

        # Streamlitボタン
        if st.button("キャッシュ(.pkl)ファイルを削除"):
            # 指定フォルダ内のpklファイルパスを取得
            pkl_files = glob.glob(os.path.join(target_folder, "*.pkl"))

            # ファイル削除処理
            for file_path in pkl_files:
                try:
                    os.remove(file_path)
                    st.write(f"削除しました: {file_path}")
                except Exception as e:
                    st.write(f"削除失敗: {file_path}, エラー: {e}")

            if not pkl_files:
                st.write("削除対象の.pklファイルはありませんでした。")

        st.divider()
        st.markdown('''# :orange[抜き出す高さ]''')
        #*関数近似を行う位置の基板からの距離[μm]
        k_extract_microm = st.number_input('基板からの距離 k_extract_microm[μm]',value=100.00,step=0.01,format="%.2f")
        #*関数近似を行う位置の基板からの距離[pix]
        debug_k_pix = st.number_input('画像下端からの距離 debug_k_pix[pix]',min_value=0, step=1, format="%d", value=500)
        radio_microm_or_pix = st.radio("μmかpixか：",("k_extract_microm[μm]", "debug_k_pix[pix]"))

        st.divider()



        st.divider()

        #! 位相分布
        st.markdown('''# :orange[位相分布]''')
        st.markdown("""## 設定""")
        #TODO 求めたい温度分布の横の長さ，バブル中心から画像横端までの長さ[pix]，0列目からNx列目までを軸対象と仮定する # 1024

        Nx = _number_input_with_session_state('バブル中心から画像左端の距離：Nx [pix]', key='Nx', default_value=512)
        #TODO 求めたい温度分布の縦の長さ[pixel] #1024
        Nz = st.number_input('温度を計算する範囲；Nz [pix]', step=1, format="%d", value = 1024, max_value=1024)
        n_apr_pix = st.number_input('位相分布を近似する範囲；n_apr_pix [pix]', step=1, format="%d", value = 700)
        h0 = st.number_input('画像下端から基板表面までの距離；h0 [pix]', step=1, format="%d", value = 200) #*画像下端から基板上面までの距離[pix]
        l = st.number_input('端のカット；l [pix]', value=20) #*l[pixel]：phseの画像を端から数えてl(エル)ピクセル目からNxピクセル目までを温度分布にして出力
        #TODO [pix]:近似を行う場合は範囲を指定(<=1024)
        #TODO [z1:z2,x1:x2]の範囲の位相を平均し，その位相を0にoffset，絶対水温の領域を指定．zは縦方向，xは横方向．順番に注意．
        x_adjust_phase = st.number_input('右を正として中心をずらす量；x_adjust [pix]', step=1, format="%d", value = 0, max_value=1024, help="offset_convolve_centeredのみで有効。0ならばoffset_convolveに対して行った近似関数の頂点の位置にずれる。さらにずらしたい量をここに入力。")
        st.markdown("""### [z1&#58;z2,x1&#58;x2]の位相を0にする""")
        z1 = st.number_input('z1[pix]', value=720)
        z2 = st.number_input('z2[pix]', value=730)
        x1 = st.number_input('x1[pix]', value=0)
        x2 = st.number_input('x2[pix]', value=20)
        #* [z1:z2,x1:x2]の範囲の位相を平均し，その位相を0にoffset，絶対水温の領域を指定．zは縦方向，xは横方向．順番に注意．
        offset_area_slice = (slice(z1, z2), slice(x1, x2)) 
        st.markdown("""### 位相計算""")
        # gaussian_additive_term = st.radio("一次関数か定数か：",("linear", "constant"), index=0)
        convolve_size_phase = _number_input_with_session_state('移動平均サイズ：convolve_size_phase', key='convolve_size_temp', default_value=21)

        show_phase_temp_dict = {
            'original': st.checkbox('original', key='fig_original'),
            'offset': st.checkbox('offset', key='fig_offset', value=True),
            'offset_convolve': st.checkbox('offset_convolve', key='fig_offset_convolve', value=True),
            'offset_convolve_centered': st.checkbox('offset_convolve_centered', key='fig_offset_convolve_centered', value=True),
            'gaussian_plus_linear':st.checkbox('gaussian_plus_linear', key='fig_gaussian_plus_linear', value=True),
            'gaussian_plus_linear_centered': st.checkbox('gaussian_plus_linear_centered', key='fig_gaussian_plus_linear_centered', value=True),
            'gaussian_plus_offset': st.checkbox('gaussian_plus_offset', key='fig_gaussian_plus_offset', value=True),
            'gaussian_plus_offset_centered': st.checkbox('gaussian_plus_offset_centered', key='fig_gaussian_plus_offset_centered', value=True),
        }
        st.markdown("""## :green[位相分布の図]""")
        st.markdown("""### 補助線など""")
        show_guide_phase_dict = {
            'extract_height (k_extract_microm or deug_k_pix)' : st.checkbox('show_phase_color_extract_height (k_extract_microm or deug_k_pix)', value=False),
            'n_apr_pix' : st.checkbox('show_phase_color_n_apr_pix', value=False),
            'subtract_surface' : st.checkbox('show_phase_color_subtract_offset (h0)', value=False),
            'phase_color_offset_region' : st.checkbox('show_phase_color_offset_region', value=False)
        }

        st.markdown("""## :green[高さ一定の位相]""")
        show_title_phase_extract = st.checkbox('title_phase_extract')

        st.divider()

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
        uniform_filter_size = st.number_input('uniform_filter_size', value=10, step=1, format="%d", help="scipyのuniform_filter1dで高さに対して移動平均を取る際のサイズ")
        min_points_required = st.number_input('min_points_required', value=10, step=1, format="%d", help="cutoffの結果この値以下の点しか得られなかった高さについては、近似を行わない。")
        
        #!共通
        st.markdown("""# あまり変更しない""")
        with st.sidebar.expander(""):
            T_room = st.number_input('室温 T_room[℃]', value=24.5) #*室温
            lamda = st.number_input('観察用レーザーの波長 lamda[μm]', value=0.532) #*観察用laser wave length [μm]
            d_micro_to_pix_temp = st.number_input('温度分布観察カメラのレート d_micro_to_pix_temp (1.9833)[pix/μm]', value=1.9833) #*温度分布観察カメラの1umあたりのpixel d[pixel/μm]
            d_micro_to_pix_flow = st.number_input('流速分布観察カメラのレート d_micro_to_pix_flow(1.0269)[pix/μm]', value=1.0269) #*流速分布観察カメラの1umあたりのpixel d[pixel/μm]

        class_commonarea_dict = {
            'fname_phase': fname_phase,
            'fname_flow': fname_flow,
            'k_extract_microm': k_extract_microm,
            'debug_k_pix': debug_k_pix,
            'radio_microm_or_pix': radio_microm_or_pix,
            'T_room': T_room,
            'Nx': Nx,
            'Nz': Nz,
            'n_apr_pix': n_apr_pix,
            'h0': h0,
            'l': l,
            'x_adjust_phase': x_adjust_phase,
            'offset_area_slice': offset_area_slice,
            'convolve_size_phase': convolve_size_phase,
            'd_micro_to_pix_temp': d_micro_to_pix_temp,
        }
        show_phase_dict = {
            
            'show_phase_temp_dict': show_phase_temp_dict,
            'show_guide_phase_dict': show_guide_phase_dict,
            'show_title_phase_extract': show_title_phase_extract
        }
        
        calc_temp_dict
        
        show_temp_dict
        
        calc_flow_dict
        
        show_flow_dict

    return (calc_phase_dict, )