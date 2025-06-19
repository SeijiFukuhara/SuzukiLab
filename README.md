# Temp_Flow_AxialSymmetry

フォルダの場所："C:\Users\seiji\Temp_Flow_AxialSymmetry"
コマンドプロンプトで streamlit run app_temp_flow_deploy.py

conda activate extract_phase_env環境での動作を確認済み

## プログラム実行の注意点
- extract_phase.py,extract_phase_fukuhara.py,extract_phase_narihira.pyは藤井先生のプログラムが基になっており，動画ファイルの冒頭のが一枚がリファレンスとして使われ，その一枚目は動画ファイルから削除されて出力される（フレーム数50の動画を入力した場合，1フレーム目の画像はリファレンスとして使われ，49フレームの動画が出力される）
- extract_phase_video.pyは入力に「リファレンス画像」「変換動画」を要求しており，「リファレンス画像」がリファレンスとして使用されるため，出力動画のフレーム数は「変換動画」と変わらない．

## 使い方
1. `extract_and_flip.py`
   -  `extract_and_flip.py path_to_video.avi start_frame end_frame --flip`で撮影した`.avi`ファイルを指定のフレームで切り抜き．`--flip`がついているときは上下逆転．省略可．
   -  出力：`.bmp`または`.avi`
2. `extract_phase_video.py`
   - `extract_phase_video.py path_to_reference.bmp path_to_video.avi`で，位相を計算フォルダを作成
   - 出力：`位相動画.avi`と`phase_csvのフォルダ`
3. `calculate_phase.py`
   - `calculate_phase.py path_to_phase.csv`で，先ほどの`phase_csvのフォルダ`に対して背景除去、移動平均を行う。
   - 出力：カラー位相の`bmpフォルダ`（フォルダ内に各フレームごとの変換画像）
4. `videoplayer.py`
   - `videoplayer.py path_to_bmpfolder1 path_to_bmpfolder2`で二つのフォルダ内のbmpを比較可能．
