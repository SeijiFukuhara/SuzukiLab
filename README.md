# Temp_Flow_AxialSymmetry

フォルダの場所："C:\Users\seiji\Temp_Flow_AxialSymmetry"
コマンドプロンプトで streamlit run app_temp_flow_deploy.py

conda activate extract_phase_env環境での動作を確認済み

## プログラム実行の注意点
- extract_phase.py,extract_phase_fukuhara.py,extract_phase_narihira.pyは藤井先生のプログラムが基になっており，動画ファイルの冒頭のが一枚がリファレンスとして使われ，その一枚目は動画ファイルから削除されて出力される（フレーム数50の動画を入力した場合，1フレーム目の画像はリファレンスとして使われ，49フレームの動画が出力される）
- extract_phase_video.pyは入力に「リファレンス画像」「変換動画」を要求しており，「リファレンス画像」がリファレンスとして使用されるため，出力動画のフレーム数は「変換動画」と変わらない．

## 使い方
1. `flip_vertically.py`
   -  `python flip_vertically.py path_to_video.avi`で撮影した`.avi`ファイルに実行して上下逆転．
2. `extract_frame.py`
   -  `extract_phase.py path_to_video.avi 1 1`でリファレンス画像を作成
   -  `extract_phase.py path_to_video.avi start_frame end_frame`で変換動画を作成．
3. `extract_phase_video.py`
   - `extract_phase_video.py path_to_reference.bmp path_to_video.avi`で，位相動画と`phase_csv`フォルダを作成
4. `calculate_phase.py`
   - `calculate_phase.py path_to_phase.csv`で，`png`と`bmp`を作成