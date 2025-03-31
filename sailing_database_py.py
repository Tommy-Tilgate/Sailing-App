import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# データクリーニング
clean_data = df.iloc[:, :12]
N = clean_data.shape[0]
for i in range(N):
    if pd.isna(clean_data.iloc[i, 0]):
        clean_data.iloc[i, 0] = clean_data.iloc[i - 1, 0]
    if clean_data.iloc[i, 8] is np.nan:
        clean_data.iloc[i, 8] = clean_data.iloc[i - 1, 8]
    if clean_data.iloc[i, 9] is np.nan:
        clean_data.iloc[i, 9] = clean_data.iloc[i - 1, 9]
    if pd.isna(clean_data.iloc[i, 5]):
        j = i - 1
        while j >= 0 and clean_data.iloc[j, 4] != clean_data.iloc[i, 4]:
            j -= 1
        if j >= 0:
            clean_data.iloc[i, 5] = clean_data.iloc[j, 5]
    if pd.isna(clean_data.iloc[i, 6]):
        j = i - 1
        while j >= 0 and clean_data.iloc[j, 4] != clean_data.iloc[i, 4]:
            j -= 1
        if j >= 0:
            clean_data.iloc[i, 6] = clean_data.iloc[j, 6]
    if pd.isna(clean_data.iloc[i, 7]):
        j = i - 1
        while j >= 0 and clean_data.iloc[j, 4] != clean_data.iloc[i, 4]:
            j -= 1
        if j >= 0:
            clean_data.iloc[i, 7] = clean_data.iloc[j, 7]

clean_data = clean_data.dropna(subset=[clean_data.columns[1]])
clean_data = clean_data.dropna(subset=[clean_data.columns[2]])

# データの個別条件ごとの平均速度を計算
condition_list = clean_data['condition'].unique()
helmsman_list = clean_data['helmsman'].unique()

personal = pd.DataFrame(columns=condition_list, index=helmsman_list)

for i, helmsman in enumerate(helmsman_list):
    for j, condition in enumerate(condition_list):
        personal.iloc[i, j] = np.average(clean_data[(clean_data['helmsman'] == helmsman) & (clean_data['condition'] == condition)]['speed'])

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
    ax.set_thetagrids(np.degrees(angles[:-1]), condition_labels, fontsize=8)
    ax.set_ylim(2, max_value)
    ax.set_title(helmsman_data.name, fontsize=10)

# 最大値を取得（全データで統一するため）
max_value = personal.max().max()

# 2×3 のグリッドで6つの図を作成
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), subplot_kw=dict(polar=True))

# 各 helmsman についてレーダーチャートを作成
for ax, helmsman in zip(axes.flat, personal.index[:3]):
    helmsman_data = personal.loc[helmsman]
    create_radar_chart(ax, helmsman_data, personal.columns, max_value)

# Streamlitでプロットを表示
st.pyplot(fig)

# 2×3 のグリッドで残りの図を作成
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), subplot_kw=dict(polar=True))

for ax, helmsman in zip(axes.flat, personal.index[4:]):
    helmsman_data = personal.loc[helmsman]
    create_radar_chart(ax, helmsman_data, personal.columns, max_value)

# Streamlitでプロットを表示
st.pyplot(fig)

# 各conditionごとに散布図と多項式回帰を描画
for condition in clean_data['condition'].unique():
    condition_data = clean_data[clean_data['condition'] == condition]

    # rakeとspeedをプロット
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='rake', y='speed', data=condition_data, s=70)
    plt.title(f'Condition: {condition}')
    plt.xlabel('Rake')
    plt.ylabel('Speed')
    plt.grid(True)

    # 2次の多項式回帰
    coeffs = np.polyfit(condition_data['rake'], condition_data['speed'], 2)
    poly = np.poly1d(coeffs)

    # 近似曲線をプロット
    x_vals = np.linspace(min(condition_data['rake']), max(condition_data['rake']), 100)
    y_vals = poly(x_vals)
    plt.plot(x_vals, y_vals, color='red', label=f'Fit: y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

    # 最大値の時のXを計算
    max_x = -coeffs[1] / (2 * coeffs[0])
    plt.axvline(x=max_x, color='green', linestyle='--', label=f'Max at X = {max_x:.2f}')

    plt.xlabel('Rake')
    plt.ylabel('Speed')
    plt.legend()
    st.pyplot(plt)  # Streamlitで表示
    plt.clf()  # 次のプロットのためにクリア

# 他の項目についても同様にプロット
for condition in clean_data['condition'].unique():
    condition_data = clean_data[clean_data['condition'] == condition]

    # bendとspeedをプロット
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='bend', y='speed', data=condition_data, s=70)
    plt.title(f'Condition: {condition}')
    plt.xlabel('Bend')
    plt.ylabel('Speed')
    plt.grid(True)

    # 2次の多項式回帰
    coeffs = np.polyfit(condition_data['bend'], condition_data['speed'], 2)
    poly = np.poly1d(coeffs)

    # 近似曲線をプロット
    x_vals = np.linspace(min(condition_data['bend']), max(condition_data['bend']), 100)
    y_vals = poly(x_vals)
    plt.plot(x_vals, y_vals, color='red', label=f'Fit: y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

    # 最大値の時のXを計算
    max_x = -coeffs[1] / (2 * coeffs[0])
    plt.axvline(x=max_x, color='green', linestyle='--', label=f'Max at X = {max_x:.2f}')

    plt.xlabel('Bend')
    plt.ylabel('Speed')
    plt.legend()
    st.pyplot(plt)  # Streamlitで表示
    plt.clf()  # 次のプロットのためにクリア

# tensionについても同様にプロット
for condition in clean_data['condition'].unique():
    condition_data = clean_data[clean_data['condition'] == condition]

    # tensionとspeedをプロット
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='tention', y='speed', data=condition_data, s=70)
    plt.title(f'Condition: {condition}')
    plt.xlabel('Tention')
    plt.ylabel('Speed')
    plt.grid(True)

    # 2次の多項式回帰
    coeffs = np.polyfit(condition_data['tention'], condition_data['speed'], 2)
    poly = np.poly1d(coeffs)

    # 近似曲線をプロット
    x_vals = np.linspace(min(condition_data['tention']), max(condition_data['tention']), 100)
    y_vals = poly(x_vals)
    plt.plot(x_vals, y_vals, color='red', label=f'Fit: y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

    # 最大値の時のXを計算
    max_x = -coeffs[1] / (2 * coeffs[0])
    st.pyplot(plt)  # Streamlitで表示
    plt.clf()  # 次のプロットのためにクリア
