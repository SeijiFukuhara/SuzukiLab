import os
import pandas as pd
import sys

# フォルダA（サブフォルダ内に .cih ファイルがある）
input_folder = sys.argv[1]  # コマンドライン引数からパスを受け取る

# 抽出したい項目（キー: .cih内の文字列, 値: CSV列名）
target_keys = {
    # "Camera Type": "Camera Type",
    # "Camera ID": "Camera ID",
    "Record Rate(fps)": "Record Rate(fps)",
    "Total Frame": "Total Frame",  # ← 追加
    "Shutter Speed(s)": "Shutter Speed",
    "Image Width": "Image Width",
    "Image Height": "Image Height",
    "Date": "Date",
    "Time": "Time",
    # "Color Type": "Color Type",
    # "File Format": "File Format",
    # "Device Name": "Device Name",
    # "IP Address": "IP Address"
}

# メタデータ格納用
all_metadata = []

# 試す文字コード一覧
encodings_to_try = ["utf-8", "shift_jis", "cp932", "iso-8859-1"]

# ディレクトリ探索
for subdir, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".cih"):
            filepath = os.path.join(subdir, file)
            metadata = {
                "Filename": file,
                "Folder": os.path.basename(subdir)
            }

            read_success = False
            for enc in encodings_to_try:
                try:
                    with open(filepath, "r", encoding=enc, errors="strict") as f:
                        for line in f:
                            if ":" in line:
                                key, value = line.split(":", 1)
                                key = key.strip()
                                value = value.strip()
                                if key in target_keys:
                                    metadata[target_keys[key]] = value
                    all_metadata.append(metadata)
                    read_success = True
                    break
                except Exception:
                    continue

            if not read_success:
                print(f"❌ 読み込み失敗（すべてのエンコーディング）: {filepath}")

# DataFrame作成
df = pd.DataFrame(all_metadata)

# 録画時間（秒）の計算（数値変換が失敗する場合に備えてNaNも許容）
if "Record Rate(fps)" in df.columns and "Total Frame" in df.columns:
    df["Recording Time (s)"] = pd.to_numeric(df["Total Frame"], errors="coerce") / pd.to_numeric(df["Record Rate(fps)"], errors="coerce")
    df["Recording Time (s)"] = df["Recording Time (s)"].fillna(0).round(2)  # NaNを0に置き換え、2桁に丸める
    df.insert(6, "Recording Time (s)", df.pop("Recording Time (s)"))

# CSVとして保存
output_path = os.path.join(input_folder, "chi_summary.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ {output_path} に保存しました。")
