# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 画像データの機械学習用
from sklearn import datasets

# JupyterNotebook上でグラフを表示する設定
get_ipython().run_line_magic('matplotlib', 'inline')
# DataFrameですべての列を表示する設定
pd.options.display.max_columns = None


# %%
digits = datasets.load_digits()


# %%
# digitsの構成
digits.keys()


# %%
digits['data'].shape


# %%
# dataから1枚の画像データを取得
temp = digits['data'][0]
# 8x8にreshape
temp = temp.reshape(8,8)
temp


# %%
# 画像化
plt.imshow(temp, cmap='Greys')


# %%
# 対応する目的変数を確認
digits['target'][0]


# %%
# 目的変数（Y)：target、説明変数（X)：data
Y = np.array(digits['target'])
X = np.array(digits['data'])


# %%
# 形状の確認
print(Y.shape)
print(X.shape)


# %%
# 必要なライブラリのインポート
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# %%
# データの分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)


# %%
# ロジスティック回帰（多クラス分類）
normal_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=3000)
normal_model.fit(X_train, Y_train)
Y_pred = normal_model.predict(X_valid)

# モデルの評価
print(classification_report(Y_valid, Y_pred))


# %%
# テストデータによる評価
Y_pred = normal_model.predict(X_test)

print(classification_report(Y_test, Y_pred))


# %%
# 学習後のパラメータを取得
coefs = normal_model.coef_

coefs.shape


# %%
# パラメータ（coef_)を画像として出力

# 2行5列にグラフを分割
fig, ax = plt.subplots(2, 5, figsize=(8, 4))

for i in range(2):
    for j in range(5):
        # zは左上から数えたグラフの描画位置
        z = i * 5 + j

        # 行=i、列=jの位置に画像を描画する
        ax[i, j].imshow(coefs[z].reshape(8,8), cmap='viridis')

        # 目盛を表示しない設定
        ax[i, j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # タイトルに数値を
        ax[i, j].set_title(z)

plt.show()


# %%
# 必要なライブラリのインポート
from sklearn.decomposition import PCA


# %%
# 64個の説明変数を8個に主成分分析
pca = PCA(n_components=8).fit(X)


# %%
# 主成分分析した結果を基にデータ変換を行い成分を取得
X2 = pca.fit_transform(X)


# %%
# 形状を確認
print("X: ", X.shape)
print("X2:", X2.shape)


# %%
# データの分割。X2を使用していることに注意
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.3, random_state=0)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)


# %%
# ロジスティック回帰（多クラス分類）
pca_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=3000)
pca_model.fit(X_train, Y_train)
Y_pred = pca_model.predict(X_valid)

# モデルの評価
print(classification_report(Y_valid, Y_pred))


# %%
# テストデータによる評価
Y_pred = pca_model.predict(X_test)

print(classification_report(Y_test, Y_pred))


# %%
# PCAのパラメータを取得
comps= pca.components_

comps.shape


# %%
# PCAのパラメータを画像として出力
fig, ax = plt.subplots(1, 8, figsize=(12, 3))

for i in range(8):
    ax[i].imshow(comps[i].reshape(8,8), cmap='viridis')
    ax[i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax[i].set_title(f"PCA{i+1}")

plt.show()


# %%
# 学習後のパラメータを取得
coefs = pca_model.coef_

coefs.shape


# %%
# パラメータ（coef_)を画像として出力

# 2行5列にグラフを分割
fig, ax = plt.subplots(2, 5, figsize=(8, 4))

for i in range(2):
    for j in range(5):
        # zは左上から数えたグラフの描画位置
        z = i * 5 + j

        # 行=i、列=jの位置に画像を描画する
        ax[i, j].imshow(coefs[z].reshape(8, 1), cmap='viridis')

        # 目盛を表示しない設定
        ax[i, j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # タイトルに数値を
        ax[i, j].set_title(z)

plt.show()


# %%
# 必要なライブラリのインポート
from sklearn.cluster import KMeans


# %%
# データの分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# %%
# クラスタリング
kmeans_model = KMeans(n_clusters=10, init='k-means++', n_init=30, random_state=0)
kmeans_model.fit(X_train)


# %%
# クラスタリング
kmeans_model = KMeans(n_clusters=10, init='k-means++', n_init=30, random_state=0)
kmeans_model.fit(X_train)


# %%
# クラスタ番号を取得
Y_pred = kmeans_model.predict(X_train)
Y_pred


# %%
# クラスタ番号を数値に変換する表の作成
import collections

corr_table = []

for i in range(10):
    # クラスタ番号i番目のY_trainを取得
    count = collections.Counter(Y_train[Y_pred == i])
    # Y_trainの中で多数決で数値を決める
    print(count.most_common())
    corr_table.append(count.most_common()[0][0])


# %%
# 同じ数値が重複した場合は、2位の数値を設定
corr_table[9] = 9
corr_table


# %%
# クラスタ番号を数値に変換
Y_pred2 = []
for i in Y_pred:
    Y_pred2.append(corr_table[i])

# モデルの評価
print(classification_report(Y_train, Y_pred2))


# %%
# クラスタ番号を取得
Y_test_pred = kmeans_model.predict(X_test)


# %%
# クラスタ番号を数値に変換
Y_test_pred2 = []
for i in Y_test_pred:
    Y_test_pred2.append(corr_table[i])

# モデルの評価
print(classification_report(Y_test, Y_test_pred2))


# %%
# 学習後のパラメータを取得
centers = kmeans_model.cluster_centers_

centers.shape


# %%
# パラメータを画像として出力

# 2行5列にグラフを分割
fig, ax = plt.subplots(2, 5, figsize=(8, 4))

for i in range(2):
    for j in range(5):
        # zは左上から数えたグラフの描画位置
        z = i * 5 + j

        # 行=i、列=jの位置に画像を描画する
        ax[i, j].imshow(centers[z].reshape(8,8), cmap='viridis')

        # 目盛を表示しない設定
        ax[i, j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # タイトルに数値を
        ax[i, j].set_title(corr_table[z])

plt.show()


# %%



