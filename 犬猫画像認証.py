#!/usr/bin/env python
# coding: utf-8

# # 【課題1】写真に映る動物が犬か猫かを分類する
# 
# ## 学習に使うデータセットをインポートして計測データと教師データに分ける

# 表示された `dc_photos` フォルダを、Cloud9のワークスペース直下（このノートブックと同じディレクトリ）にアップロードします。
# アップロードが完了した状態で、下記のコードを実行して、画像のデータセットを読み込んでください。

# In[1]:


import imageio
import numpy as np

# 写真は全て 75ピクセル × 75ピクセル のRGBカラー画像
PHOTO_SIZE = 75 * 75 * 3

# 空の配列（計測データ X と教師データ y）を用意する
#empty関数で初期化・uint8(符号なし8bit整数で表現)
X = np.empty((0, PHOTO_SIZE), np.uint8)
y = np.empty(0, np.uint8)

# 犬と猫の画像を配列形式で読み込んでXに格納（axis = 0で二次元配列の縦（行）に要素を追加する）
# y には 犬 なら 0, 猫 なら 1 を割り当て、整数値のデータをappend関数で配列に追加
#03d は、３桁以下の時に0で詰める。例えば15の時は015と表示される。
for i in range(1, 201):
    p1 = imageio.imread(f"dc_photos/dogs/dog-{i:03d}.jpg").reshape(1, PHOTO_SIZE)
    X = np.append(X, p1, axis = 0)
    y = np.append(y, np.array([0], dtype = np.uint8))
    
    p2 = imageio.imread(f"dc_photos/cats/cat-{i:03d}.jpg").reshape(1, PHOTO_SIZE)
    X = np.append(X, p2, axis = 0)
    y = np.append(y, np.array([1], dtype = np.uint8))


# `X` と `y` の要素数を確認してください。

# In[2]:


# Xおよびyの要素数を確認する（命令を追記すること）
print(X)
print(y)


# Xとyの中身の確認は、要素数が多いので省略。上記の繰り返し処理により、Xとyには、犬と猫の情報が交互に格納された状態になっています。また、犬と猫それぞれの写真には偏りがありません。（実際のデータを確認してみてください。Jupyter Notebookに表示する処理は省略します。）
# 
# ## データを訓練データとテストデータに分ける
# 
# 今回の課題において `train_test_split` は使用せず、単純に300件目で区切る形で訓練データとテストデータを分けましょう。

# In[3]:


X_train = X[:301]
X_test  = X[301:]
y_train = y[:301]
y_test  = y[301:]


# Xの全ての要素は 0 から 255までのデータ（色の強さを255段階で表現したもの）になっていますので、スケーリングも不要です。
# 
# ## 訓練データを用いて分類器を作成する
# 
# このまま線形分類器を作成して、訓練データを設定しましょう。

# In[4]:


# 線形分類器を作成して訓練データを設定する（命令を追記すること）
from sklearn.svm import SVC
classifier = SVC(kernel = "linear")
classifier.fit(X_train, y_train)


# ## テストデータを分類器にかけて分類を実施する
# 
# テストデータを分類器にかけてください。

# In[5]:


## テストデータを分類器にかける（命令を追記すること）
y_pred = classifier.predict(X_test)


# ## 結果を表示する
# 
# では、分類器が出力した結果と正解を比較してみましょう。

# In[6]:


# 分類器の出力結果を表示する（命令を追記すること）
print(y_pred)


# In[7]:


# 正解を表示する（命令を追記すること）
print(y_test)


# In[8]:


# 混合行列を表示する（命令を追記すること）
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))


# In[9]:


# 正答率を確認する（命令を追記すること）
print(metrics.classification_report(y_test, y_pred))


# In[ ]:




