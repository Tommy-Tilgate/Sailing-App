# -*- coding: utf-8 -*-
"""Sailing Database Streamlit

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11y_5spOG7QzredUhvjuAtfPJAFfMv-VQ
"""

import streamlit as st
import pandas as pd
import subprocess
import sys

# タイトル
st.title("Sailing Database Web App")

# ファイルアップロード
uploaded_file = st.file_uploader("ファイルをアップロードしてください", type=["csv", "xlsx"])

if uploaded_file is not None:
    # ファイルの種類を判定
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # データの表示
    st.write("アップロードされたデータ:")
    st.dataframe(df)

    # スクリプトを実行し、出力を取得
    with open("temp_input.csv", "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)

    result = subprocess.run([sys.executable, "sailing_database_py.py"], capture_output=True, text=True)

    # スクリプトの出力を表示
    st.write("スクリプトの出力:")
    st.text(result.stdout)

    # エラーがある場合は表示
    if result.stderr:
        st.error("エラーが発生しました:")
        st.text(result.stderr)