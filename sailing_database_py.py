# -*- coding: utf-8 -*-
"""sailing_database.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cG1fNMXCgYUybhV4WguADhtuEQj7nlI5
"""

import japanize_matplotlib
import seaborn as sns
sns.set(font='IPAexGothic')

import pandas as pd
import streamlit as st

st.title("スプレッドシート解析アプリ")
st.write("CSVまたはExcelファイルをアップロードしてください。")

# ファイルアップロード
data = st.file_uploader("ファイルを選択", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # ファイルの拡張子を確認して読み込む
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # データの基本情報を表示
    st.subheader("データの基本情報")
    st.write(df.info())

    # データの先頭5行を表示
    st.subheader("データの先頭5行")
    st.write(df.head())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



clean_data=data.iloc[:,:12]
N=clean_data.shape[0]
for i in range(N):
  if pd.isna(clean_data.iloc[i, 0]):
        clean_data.iloc[i, 0] = clean_data.iloc[i - 1, 0]
  if clean_data.iloc[i,8] is np.nan:
    clean_data.iloc[i,8]=clean_data.iloc[i-1,8]
  if clean_data.iloc[i,9] is np.nan:
    clean_data.iloc[i,9]=clean_data.iloc[i-1,9]# Check if current value is NaN
        # Check if the value in the other column matches the previous row's value
  if pd.isna(clean_data.iloc[i, 5]):  # Check if the current value is NaN
    j = i - 1  # Start looking at the previous row
        # Continue moving back until we find a row with the same value in 'other_column'
    while j >= 0 and clean_data.iloc[j,4] != clean_data.iloc[i, 4]:
            j -= 1
    if j >= 0:  # If we found a match
            clean_data.iloc[i, 5] = clean_data.iloc[j, 5]  # Replace with the value from the found row
  if pd.isna(clean_data.iloc[i, 6]):  # Check if the current value is NaN
    j = i - 1  # Start looking at the previous row
        # Continue moving back until we find a row with the same value in 'other_column'
    while j >= 0 and clean_data.iloc[j,4] != clean_data.iloc[i, 4]:
            j -= 1
    if j >= 0:  # If we found a match
            clean_data.iloc[i, 6] = clean_data.iloc[j, 6]
  if pd.isna(clean_data.iloc[i, 7]):  # Check if the current value is NaN
    j = i - 1  # Start looking at the previous row
        # Continue moving back until we find a row with the same value in 'other_column'
    while j >= 0 and clean_data.iloc[j,4] != clean_data.iloc[i, 4]:
            j -= 1
    if j >= 0:  # If we found a match
            clean_data.iloc[i, 7] = clean_data.iloc[j, 7]

clean_data=clean_data.dropna(subset=[clean_data.columns[1]])
clean_data=clean_data.dropna(subset=[clean_data.columns[2]])
print(clean_data)

condition_list=clean_data['condition'].unique()
helmsman_list=clean_data['helmsman'].unique()



personal=pd.DataFrame(columns=condition_list,index=helmsman_list)



for i,helmsman in enumerate(helmsman_list):
  for j,condition in enumerate(condition_list):
    personal.iloc[i,j]=np.average(clean_data[(clean_data['helmsman']==helmsman)&(clean_data['condition']==condition)]['speed'])

import matplotlib.pyplot as plt
import numpy as np

# レーダーチャートの描画関数
def create_radar_chart(ax, helmsman_data, condition_labels, max_value):
    helmsman_data = helmsman_data.fillna(0)

    angles = np.linspace(0, 2 * np.pi, len(condition_labels), endpoint=False).tolist()
    values = helmsman_data.values.tolist()

    # 閉じた図形にするために最初の点を追加
    angles.append(angles[0])
    values.append(values[0])

    # プロット
    ax.plot(angles, values, marker='o')
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), condition_labels, fontsize=8)  # ラベルのサイズを小さく
    ax.set_ylim(2,max_value)  # すべての図で同じスケール
    ax.set_title(helmsman_data.name, fontsize=10)

# 最大値を取得（全データで統一するため）
max_value = personal.max().max()

# 2×3 のグリッドで6つの図を作成
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), subplot_kw=dict(polar=True))

# 各 helmsman についてレーダーチャートを作成
for ax, helmsman in zip(axes.flat, personal.index[:3]):
    helmsman_data = personal.loc[helmsman]
    create_radar_chart(ax, helmsman_data, personal.columns, max_value)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# レーダーチャートの描画関数
def create_radar_chart(ax, helmsman_data, condition_labels, max_value):
    helmsman_data = helmsman_data.fillna(0)

    angles = np.linspace(0, 2 * np.pi, len(condition_labels), endpoint=False).tolist()
    values = helmsman_data.values.tolist()

    # 閉じた図形にするために最初の点を追加
    angles.append(angles[0])
    values.append(values[0])

    # プロット
    ax.plot(angles, values, marker='o')
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), condition_labels, fontsize=8)  # ラベルのサイズを小さく
    ax.set_ylim(min_value,max_value)  # すべての図で同じスケール
    ax.set_title(helmsman_data.name, fontsize=10)

# 最大値を取得（全データで統一するため）
max_value = personal.max().max()
min_value = personal.min().min()

# 2×3 のグリッドで6つの図を作成
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), subplot_kw=dict(polar=True))

# 各 helmsman についてレーダーチャートを作成
for ax, helmsman in zip(axes.flat, personal.index[4:]):
    helmsman_data = personal.loc[helmsman]
    create_radar_chart(ax, helmsman_data, personal.columns, max_value)

plt.tight_layout()
plt.show()

for condition in clean_data['condition'].unique():
    condition_data = clean_data[clean_data['condition'] == condition]

    # rakeとspeedをプロット
    plt.figure(figsize=(4,3))  # 図のサイズを指定
    sns.scatterplot(x='rake', y='speed', data=condition_data, s=70)
    plt.title(f'Condition: {condition}')  # タイトルにconditionを表示
    plt.xlabel('Rake')  # x軸ラベル
    plt.ylabel('Speed')  # y軸ラベル
    plt.grid(True)

    # 2次の多項式回帰
    coeffs = np.polyfit(condition_data['rake'], condition_data['speed'], 2)
    poly = np.poly1d(coeffs)  # 多項式関数を作成

    # 近似曲線をプロット
    x_vals = np.linspace(min(condition_data['rake']), max(condition_data['rake']), 100)
    y_vals = poly(x_vals)
    plt.plot(x_vals, y_vals, color='red', label=f'Fit: y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

    # 最大値の時のXを計算（2次関数の頂点）
    max_x = -coeffs[1] / (2 * coeffs[0])  # 頂点のx座標
    plt.axvline(x=max_x, color='green', linestyle='--', label=f'Max at X = {max_x:.2f}')  # 最大値のXに垂直ラインを引く

    plt.xlabel('Rake')  # x軸ラベル
    plt.ylabel('Speed')  # y軸ラベル
    plt.legend()  # グリッドを表示
    plt.show()

for condition in clean_data['condition'].unique():
    condition_data = clean_data[clean_data['condition'] == condition]

    # rakeとspeedをプロット
    plt.figure(figsize=(4,3))  # 図のサイズを指定
    sns.scatterplot(x='bend', y='speed', data=condition_data, s=70)
    plt.title(f'Condition: {condition}')  # タイトルにconditionを表示
    plt.xlabel('bend')  # x軸ラベル
    plt.ylabel('Speed')  # y軸ラベル
    plt.grid(True)

    # 2次の多項式回帰
    coeffs = np.polyfit(condition_data['bend'], condition_data['speed'], 2)
    poly = np.poly1d(coeffs)  # 多項式関数を作成

    # 近似曲線をプロット
    x_vals = np.linspace(min(condition_data['bend']), max(condition_data['bend']), 100)
    y_vals = poly(x_vals)
    plt.plot(x_vals, y_vals, color='red', label=f'Fit: y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

    # 最大値の時のXを計算（2次関数の頂点）
    max_x = -coeffs[1] / (2 * coeffs[0])  # 頂点のx座標
    plt.axvline(x=max_x, color='green', linestyle='--', label=f'Max at X = {max_x:.2f}')  # 最大値のXに垂直ラインを引く

    plt.xlabel('bend')  # x軸ラベル
    plt.ylabel('Speed')  # y軸ラベル
    plt.legend()  # グリッドを表示
    plt.show()

for condition in clean_data['condition'].unique():
    condition_data = clean_data[clean_data['condition'] == condition]

    # rakeとspeedをプロット
    plt.figure(figsize=(4,3))  # 図のサイズを指定
    sns.scatterplot(x='tention', y='speed', data=condition_data, s=70)
    plt.title(f'Condition: {condition}')  # タイトルにconditionを表示
    plt.xlabel('Rake')  # x軸ラベル
    plt.ylabel('Speed')  # y軸ラベル
    plt.grid(True)

    # 2次の多項式回帰
    coeffs = np.polyfit(condition_data['tention'], condition_data['speed'], 2)
    poly = np.poly1d(coeffs)  # 多項式関数を作成

    # 近似曲線をプロット
    x_vals = np.linspace(min(condition_data['tention']), max(condition_data['tention']), 100)
    y_vals = poly(x_vals)
    plt.plot(x_vals, y_vals, color='red', label=f'Fit: y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

    # 最大値の時のXを計算（2次関数の頂点）
    max_x = -coeffs[1] / (2 * coeffs[0])  # 頂点のx座標
    plt.axvline(x=max_x, color='green', linestyle='--', label=f'Max at X = {max_x:.2f}')  # 最大値のXに垂直ラインを引く

    plt.xlabel('tention')  # x軸ラベル
    plt.ylabel('Speed')  # y軸ラベル
    plt.legend()  # グリッドを表示
    plt.show()
