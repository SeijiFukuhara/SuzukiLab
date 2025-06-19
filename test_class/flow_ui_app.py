# flow_ui_app.py

import streamlit as st
from flow_analyzer import FlowAnalyzer

# Streamlit UI layout
def main():
    st.title("流速解析アプリ")

    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

    if uploaded_file is not None:
        d_micro_to_pix = st.number_input("1ピクセルあたりのマイクロメートル長", value=1.0)
        r_min = st.number_input("最小半径 [pix]", value=10.0)
        r_max = st.number_input("最大半径 [pix]", value=100.0)

        # アナライザーを初期化
        analyzer = FlowAnalyzer(uploaded_file, d_micro_to_pix)

        if st.button("解析を実行"):
            theta_vr_dict, number = analyzer.compute_theta_vr(r_min, r_max)
            st.write(f"取得したデータ数: {number}")
            
            st.subheader("流速グラフ (vr vs θ)")
            analyzer.plot_flow()

            st.subheader("近似曲線")
            analyzer.plot_approximation()

            if st.button("テキスト出力"):
                txt = analyzer.generate_report(r_min, r_max, number)
                st.text(txt)

if __name__ == "__main__":
    main()
