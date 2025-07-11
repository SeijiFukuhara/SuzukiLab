import streamlit as st

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

def render_sidebar_phase():
    with st.sidebar:
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
        x_adjust = st.number_input('右を正として中心をずらす量；x_adjust [pix]', step=1, format="%d", value = 0, max_value=1024, help="offset_convolve_centeredのみで有効。0ならばoffset_convolveに対して行った近似関数の頂点の位置にずれる。さらにずらしたい量をここに入力。")
        st.markdown("""### [z1&#58;z2,x1&#58;x2]の位相を0にする""")
        z1 = st.number_input('z1[pix]', value=720)
        z2 = st.number_input('z2[pix]', value=730)
        x1 = st.number_input('x1[pix]', value=0)
        x2 = st.number_input('x2[pix]', value=20)
        st.markdown("""### 位相計算""")
        # gaussian_additive_term = st.radio("一次関数か定数か：",("linear", "constant"), index=0)
        convolve_size_temp = _number_input_with_session_state('移動平均サイズ：convolve_size_temp', key='convolve_size_temp', default_value=21)

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

    return (
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
        show_title_phase_extract
    )