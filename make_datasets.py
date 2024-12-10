import os
import pickle
import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    """
    カテゴリ分類用のデータジェネレーター。
    シンプルに単一アイテム単位で返す。
    """
    def __init__(self, data_dir, batch_size=32, shuffle=True, mode='train_category'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self._load_data()
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_data(self):
        data_path = os.path.join(self.data_dir, f'{self.mode}.pkl')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found.")
        with open(data_path, 'rb') as f:
            self.X, self.y = pickle.load(f)
        self.y = np.array(self.y)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indexes]
        y_batch = self.y[batch_indexes]
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.float32)
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class SetMatchingDataGenerator(tf.keras.utils.Sequence):
    """
    SetMatchingモデル用のデータジェネレーター。
    (X1, X2), y形式。
    """
    def __init__(self, data_dir, batch_size=32, shuffle=True, mode='train'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self._load_data()
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_data(self):
        path = os.path.join(self.data_dir, f'set_matching_{self.mode}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        with open(path, 'rb') as f:
            X1, X2, y = pickle.load(f)
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.y = np.array(self.y)

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X1_batch = self.X1[batch_indexes]
        X2_batch = self.X2[batch_indexes]
        y_batch = self.y[batch_indexes]
        return (X1_batch, X2_batch), y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class trainDataGenerator(tf.keras.utils.Sequence):
    """
    Shift15mのようなセット単位のデータを返すデータジェネレーター。
    train_category.pklなどからX,yを読み込み、
    max_item_num毎にアイテムをグルーピングしてセットを形成。
    """

    def __init__(self, data_dir='uncompressed_data', batch_size=32, max_item_num=5, shuffle=True, mode='train'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_item_num = max_item_num
        self.shuffle = shuffle
        self.mode = mode
        self._load_data()
        self._make_sets()

        self.indexes = np.arange(len(self.set_Y))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_data(self):
        # ここではtrain_category.pklを利用（カテゴリラベルでグルーピング）
        if self.mode == 'train':
            mode_file = 'train_category.pkl'
        elif self.mode == 'validation':
            mode_file = 'validation_category.pkl'
        else:
            mode_file = 'test_category.pkl'

        data_path = os.path.join(self.data_dir, mode_file)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found.")
        with open(data_path,'rb') as f:
            self.X, self.y = pickle.load(f)  # X: (N,feat_dim=128), y:(N,)
        self.X = np.array(self.X,dtype=np.float32)
        self.y = np.array(self.y,dtype=np.float32)

    def _make_sets(self):
        # N個のアイテムをmax_item_num個ずつセットにまとめる
        N = len(self.X)
        # Nがmax_item_numで割り切れない場合カット
        M = (N // self.max_item_num)*self.max_item_num
        self.X = self.X[:M]
        self.y = self.y[:M]

        # reshape (M, feat_dim=128) -> (M/max_item_num, max_item_num, feat_dim=128)
        feat_dim = self.X.shape[1]
        self.set_X = self.X.reshape(-1, self.max_item_num, feat_dim)
        # setラベルはアイテムラベルの先頭を代表ラベルとする
        self.set_Y = self.y.reshape(-1, self.max_item_num)
        self.set_Y = self.set_Y[:,0] # 各セットの先頭アイテムラベルをセットラベルとする

        # x_sizeは全てmax_item_num
        self.x_size = np.full((len(self.set_Y),), self.max_item_num, dtype=np.float32)

    def __len__(self):
        return int(np.ceil(len(self.set_Y)/self.batch_size))

    def __getitem__(self,index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.set_X[batch_indexes]
        Y_batch = self.set_Y[batch_indexes]
        x_size_batch = self.x_size[batch_indexes]
        return (X_batch, x_size_batch), Y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation_val(self):
        # validation用、trainでなくvalidationのとき呼ばれる想定
        # 今はtrainDataGeneratorでvalidationするために別途インスタンス化するか、
        # ここでは別メソッドでバリデーション用データを同様に生成

        # validationデータロード
        val_path = os.path.join(self.data_dir, 'validation_category.pkl')
        if not os.path.exists(val_path):
            # validationないならtrainと同じデータ返すなど暫定対応
            return self.set_X, self.x_size, self.set_Y

        with open(val_path, 'rb') as f:
            X_val, y_val = pickle.load(f)
        X_val = np.array(X_val,dtype=np.float32)
        y_val = np.array(y_val,dtype=np.float32)

        N = len(X_val)
        M = (N//self.max_item_num)*self.max_item_num
        X_val = X_val[:M]
        y_val = y_val[:M]
        feat_dim = X_val.shape[1]
        X_val_set = X_val.reshape(-1, self.max_item_num, feat_dim)
        y_val_set = y_val.reshape(-1, self.max_item_num)[:,0]
        x_size_val = np.full((len(y_val_set),), self.max_item_num, dtype=np.float32)

        return X_val_set, x_size_val, y_val_set
