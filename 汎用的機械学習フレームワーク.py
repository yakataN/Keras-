# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
# # 機械学習に関するあれこれ
# 
# 1. Data収集
# 2. 前処理と整形
# 3. モデルクラス作成
#     1. 形をつくる
#     2. trainを作る
#     3. saveを作る
#     4. ログを作る
# 4. 学習コードを書く。
# 
# 処理の順番はこのように描くが、実際には３→１→２→４→の順で上から書く。
# 
# 
# 

#%%
from google.colab import drive
drive.mount('/content/drive')
# now at ~/content
get_ipython().run_line_magic('cd', '/content/drive/My\\ Drive/Projects/004_make_DL_with_keras/')
get_ipython().system('pwd')


#%%
get_ipython().system('pip install -q tf-nightly-2.0-preview')
# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


#%%
# import群
"""
使うレイヤー群を書く。
Dense, Conv2D, Activation, BatchNormalization, Inputあたりはとりあえず使う。
場合によってはConcatenate, AveragePooling2D, GlobalAveragePooling2Dなども。
"""
from keras.layers import Dense, Conv2D, Activation, BatchNormalization, Input, MaxPooling2D, Flatten
from keras.models import Model
# だいたいAdam. 最近ではQHAdamも気になる
from keras.optimizers import Adam
from keras.utils import to_categorical
# データ水増しを行う場合ここのチェックを外す
# from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 外から持ってくることもあるが、内から使うときはこれ。"cifar10", "mnist"など
# mnistはあとでresizeをする必要がある（channel次元を導入する）ので注意
from keras.datasets import cifar10
from keras.callbacks import TensorBoard


#%%
class Net:
    def __init__(self, batch_size=128, epochs=20):
        # 必要なインスタンス変数を指定する
        self.model = self.create_model()
        self.epochs = epochs
        self.batch_size = batch_size

    def create_model(self):
        # cifar10用になっているが適宜変更
        height = 32
        width = 32
        ch = 3
        input = Input(shape=(height, width, ch))

        # ここにlayer群を書く
        """
        # example
        n = 16
        x = Conv2D(n, (1,1))(input)
        output = Dense(10, activation="softmax")(x)
        """

        model = Model(input, output)
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        # compile
        # model は出力まで行ったので、それをどうやって評価して逆伝播させるかはここで指定する
        self.model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["acc"])

        """
        # ここでcallbackを指定する
        https://keras.io/ja/callbacks/ にあるものをlistにして渡せばいいらしい。
        """
        tensorboard = TensorBoard()
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, callbacks=[tensorboard], epochs=self.epochs, validation_data=(X_val, y_val))
        
        """
        - save
        source(2019/08/28 15:32):
        https://keras.io/ja/getting-started/faq/
        save:
            self.model.save('./models/my_model.h5')
        load:
            model = load_model('my_model.h5')
        """
        # 名前は変更すること。
        filename = "my_model.h5"
        save.model.save('/models/{}'.format(filename))
        
        return history


#%%
if __name__ == "__main__":
    net = Net()
    net.model.summary()

    # - input data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = (X_train / 255.0).astype("float32")
    X_test = (X_train / 255.0).astype("float32")
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")


#%%
if __name__ == '__main__'
    hist = net.train(X_train, y_train, X_test, y_test)
    print("history:")
    print(hist.history)


#%%
get_ipython().run_line_magic('tensorboard', '--logdir=./logs')


