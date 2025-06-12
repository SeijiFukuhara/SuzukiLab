# Temp_Flow_AxialSymmetry

フォルダの場所："C:\Users\seiji\Temp_Flow_AxialSymmetry"
コマンドプロンプトで streamlit run app_temp_flow_deploy.py

conda activate extract_phase_env環境での動作を確認済み

## プログラム実行の注意点
- extract_phase.py,extract_phase_fukuhara.py,extract_phase_narihira.pyは藤井先生のプログラムが基になっており，動画ファイルの冒頭のが一枚がリファレンスとして使われ，その一枚目は動画ファイルから削除されて出力される（フレーム数50の動画を入力した場合，1フレーム目の画像はリファレンスとして使われ，49フレームの動画が出力される）
- extract_phase_video.pyは入力に「リファレンス画像」「変換動画」を要求しており，「リファレンス画像」がリファレンスとして使用されるため，出力動画のフレーム数は「変換動画」と変わらない．

