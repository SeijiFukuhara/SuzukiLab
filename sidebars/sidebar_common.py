import streamlit as st
import os
import glob

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
            pkl_files = glob.glob(os.path.join(target, "*.pkl"))

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




        #!共通
        st.markdown("""# あまり変更しない""")
        with st.sidebar.expander(""):
            T_room = st.number_input('室温 T_room[℃]', value=24.5) #*室温
            lamda = st.number_input('観察用レーザーの波長 lamda[μm]', value=0.532) #*観察用laser wave length [μm]
            d_micro_to_pix_temp = st.number_input('温度分布観察カメラのレート d_micro_to_pix_temp (1.9833)[pix/μm]', value=1.9833) #*温度分布観察カメラの1umあたりのpixel d[pixel/μm]
            d_micro_to_pix_flow = st.number_input('流速分布観察カメラのレート d_micro_to_pix_flow(1.0269)[pix/μm]', value=1.0269) #*流速分布観察カメラの1umあたりのpixel d[pixel/μm]

        st.divider()

    return fname_phase, fname_flow, k_extract_microm, debug_k_pix, radio_microm_or_pix, T_room, lamda, d_micro_to_pix_temp, d_micro_to_pix_flow