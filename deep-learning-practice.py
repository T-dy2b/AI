# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# #  ディープラーニング
# 
# 今回の課題では手書き文字の認識をCNNを用いて行います。
# 
# 下記にKerasから手書き文字のデータセットをダウンロードするコードが記載されています。
# 
# このデータを用いてディープラーニングのモデルを構築してください。
# 
# 今までのレッスンで学んだ内容を踏まえ、各セルに'#コメント'の内容を実行するコードを記入してください。
# 
# ※既にソースコードが記載されているセルは変更不要です。
# %% [markdown]
# ## 1. ライブラリのインポート

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Keras
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# データの分割
from sklearn.model_selection import train_test_split

# 手書き数字のデータセット
from tensorflow.keras.datasets import mnist

# JupyterNotebook上でグラフを表示する設定
get_ipython().run_line_magic('matplotlib', 'inline')
# DataFrameで全ての列を表示する設定
pd.options.display.max_columns = None

# %% [markdown]
# ## 2. データの読込
# Kerasのデータセットは予めTraining setとTest setに分けられています。戻り値はタプルで取得できます。

# %%
# Kerasに添付されている手書き数字のデータセットをダウンロード
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# %% [markdown]
# ## 3.データの確認

# %%
# 形状の確認
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# %%
# X_trainの先頭1行を表示
print(X_train[:0])


# %%
# Y_trainの先頭1行を表示
print(Y_train[:0])

# %% [markdown]
# ### 手書き数字の可視化

# %%
# 「数字:空のリスト」の辞書を作成する
images = {label: [] for label in range(0,10)}


# %%
# 総イメージ数
image_count = 0

# それぞれの数字のリストに、説明変数をappendしていく
for i in range(0, len(X_train)):
    if len(images[Y_train[i]]) < 10:
        images[Y_train[i]].append(X_train[i])
        image_count += 1
        if image_count == 100:
            break


# %%
# 少し時間がかかります。
# 10行10列にグラフを分割
fig, ax = plt.subplots(10, 10, figsize=(5, 5))

for i in range(10):
    # ラベル
    ax[i, 0].set_ylabel(i)

    for j in range(10):
        # zは左上から数えたグラフの描画位置
        z = i * 10 + j

        # 行=i、列=jの位置に画像を描画する
        ax[i, j].imshow(images[i][j].reshape(28, 28), cmap='Greys')

        # 目盛を表示しない設定
        ax[i, j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

plt.show()

# %% [markdown]
# ## 4. データの前処理

# %%
# 形状の確認
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# %%
# len関数を使い、X_trainを(X_trainの長さ, 28, 28, 1)にreshapeしてX_train2に代入
X_train2 = X_train.reshape(len(X_train), 28, 28, 1)

# len関数を使い、X_testを(X_testの長さ, 28, 28, 1)にreshapeしてX_test2に代入
X_test2 = X_test.reshape(len(X_test), 28, 28, 1)


# %%
# to_categoricalを使い、Y_trainをカテゴリー変数に展開してY_train2に代入
Y_train2 = to_categorical(Y_train)

# to_categoricalを使い、Y_testをカテゴリー変数に展開してY_test2に代入
Y_test2 = to_categorical(Y_test)


# %%
# 形状の確認
print("Y_train2=", Y_train2.shape, ", X_train2=", X_train2.shape)
print("Y_test2=", Y_test2.shape, ", X_test2=", X_test2.shape)


# %%
# train_test_splitを使いデータを7:3に分割
# 機械学習用データ(X_train2、Y_train2)を「X_train2, X_valid2, Y_train2, Y_valid2」に分割
X_train2, X_valid2, Y_train2, Y_valid2 = train_test_split(X_train2, Y_train2, test_size=0.3, random_state=0)


# %%
# データ(学習、検証、テスト)の形状を確認
print("Y_train2=", Y_train2.shape, ", X_train2=", X_train2.shape)
print("Y_valid2=", Y_valid2.shape, ", X_valid2=", X_valid2.shape)
print("Y_test2=", Y_test2.shape, ", X_test2=", X_test2.shape)

# %% [markdown]
# ## 5. モデルの構築
# 
# Kerasを使ってモデルを構築してみましょう。以下を条件とします
# 
# - CNN(Conv2D)を使うこと
# - 正解率(accuracy)が50%以上であること
# 
# 場合によっては、学習にものすごく時間がかかる場合もあります。適宜パラメータ数を調整して行ってください

# %%
# ライブラリのインポート
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers


# %%
# モデルの初期化
model = keras.Sequential()

# ここにモデルを構築するコードを記述してください
model.add(Conv2D(8, kernel_size=3, padding="same", strides=1,
    input_shape=(28, 28, 1,), activation="relu"))
model.add(Flatten())
# 隠れ層
model.add(Dense(16, activation='relu'))

# 出力層
model.add(Dense(10, activation='softmax'))

# モデルの構築
model.compile(optimizer = "rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])


# %%
# モデルの構造を表示

model.summary()


# %%
get_ipython().run_cell_magic('time', '', "# 学習を実施し、結果をlogで受け取る。EarlyStoppingを使用する\nlog = model.fit(X_train2, Y_train2, epochs=5, batch_size=32, verbose=True,\n                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',\n                                                         min_delta=0, patience=100,\n                                                         verbose=1)],\n         validation_data=(X_valid2, Y_valid2))\n")


# %%
# 学習の課程をグラフで表示する
plt.plot(log.history['loss'], label='loss')
plt.plot(log.history['val_loss'], label='val_loss')
plt.legend(frameon=False) # 凡例の表示
plt.xlabel("epochs")
plt.ylabel("crossentropy")
plt.show()

# %% [markdown]
# ## 6. テストデータによる評価

# %%
# 環境により、そのままX_test2を使うとエラーになる対策(float型に変換)
X_test2 = X_test2 * 1.0


# %%
# predict_classesを使い、X_test2をもとに予測した結果をY_pred2に代入
Y_pred2 = model.predict_classes(X_test2)


# %%
# カテゴリー変数Y_test2を復元してY_test2_に代入
Y_test2_ = np.argmax(Y_test2, axis=1)


# %%
# classification_reportを使い、モデルの評価を実施
from sklearn.metrics import classification_report
print(classification_report(Y_test2_, Y_pred2))


# %%



