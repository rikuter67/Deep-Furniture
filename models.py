import tensorflow as tf
import numpy as np
import psutil
import os
import sys
from util import SetAccuracy  # カスタムメトリクスをインポート
import pdb

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, size_d, epsilon=1e-3, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.size_d = size_d

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(self.size_d,), initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(self.size_d,), initializer='zeros', trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        norm = (x - mean) / (std + self.epsilon)
        output = self.gamma * norm + self.beta
        return output

class SetAttention(tf.keras.layers.Layer):
    def __init__(self, head_size=64, num_heads=2, **kwargs):
        super(SetAttention, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.linearQ = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)
        self.linearK = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)
        self.linearV = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)
        self.linearH = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)

    def build(self, input_shape):
        super(SetAttention, self).build(input_shape)

    def call(self, q, k):
        batch_size = tf.shape(q)[0]

        q_ = self.linearQ(q)
        k_ = self.linearK(k)
        v_ = self.linearV(k)

        q_ = tf.reshape(q_, [batch_size, -1, self.num_heads, self.head_size])
        k_ = tf.reshape(k_, [batch_size, -1, self.num_heads, self.head_size])
        v_ = tf.reshape(v_, [batch_size, -1, self.num_heads, self.head_size])

        q_ = tf.transpose(q_, [0, 2, 1, 3])
        k_ = tf.transpose(k_, [0, 2, 1, 3])
        v_ = tf.transpose(v_, [0, 2, 1, 3])

        score = tf.matmul(q_, k_, transpose_b=True) / tf.sqrt(tf.cast(self.head_size, tf.float32))
        weights = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(weights, v_)

        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, -1, self.head_size * self.num_heads])
        output = self.linearH(output)
        return output

class CrossSetScore(tf.keras.layers.Layer):
    def __init__(self, head_size=64, num_heads=2, **kwargs):
        super(CrossSetScore, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.linear_gallery = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)
        self.linear_query = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)

    def build(self, input_shape):
        super(CrossSetScore, self).build(input_shape)

    def call(self, inputs):
        gallery, query = inputs
        gallery = self.linear_gallery(gallery)
        query = self.linear_query(query)

        gallery = tf.reduce_mean(gallery, axis=1)
        query = tf.reduce_mean(query, axis=1)

        sim_matrix = tf.matmul(gallery, query, transpose_b=True)
        return sim_matrix

class SMN(tf.keras.Model):
    def __init__(self, isCNN=True, seed_init=None, num_layers=1, num_heads=2, rep_vec_num=1, dim=128, **kwargs):
        super(SMN, self).__init__(**kwargs)
        self.isCNN = isCNN
        self.num_layers = num_layers
        self.rep_vec_num = rep_vec_num
        self.dim = dim
        self.seed_init = seed_init

        if self.isCNN:
            self.fc_cnn_proj = tf.keras.layers.Dense(self.dim, use_bias=False)
        else:
            self.fc_cnn_proj = None

        # self.set_emb = self.add_weight(name='set_emb', shape=(1, self.rep_vec_num, self.dim), initializer='zeros', trainable=True)

        self.self_attentionsX = [SetAttention(head_size=64, num_heads=num_heads) for _ in range(num_layers)]
        self.layer_norms_enc1X = [LayerNormalization(size_d=self.dim) for _ in range(num_layers)]
        self.layer_norms_enc2X = [LayerNormalization(size_d=self.dim) for _ in range(num_layers)]
        self.fcs_encX = [tf.keras.layers.Dense(self.dim, use_bias=False) for _ in range(num_layers)]

        self.cross_attentions = [SetAttention(head_size=64, num_heads=num_heads) for _ in range(num_layers)]
        self.layer_norms_dec1 = [LayerNormalization(size_d=self.dim) for _ in range(num_layers)]
        self.layer_norms_dec2 = [LayerNormalization(size_d=self.dim) for _ in range(num_layers)]
        self.layer_norms_decq = [LayerNormalization(size_d=self.dim) for _ in range(num_layers)]
        self.layer_norms_deck = [LayerNormalization(size_d=self.dim) for _ in range(num_layers)]
        self.fcs_dec = [tf.keras.layers.Dense(self.dim, use_bias=False) for _ in range(num_layers)]

        self.cross_set_score = CrossSetScore(head_size=64, num_heads=num_heads)

    def call(self, inputs, training=False):
        X, y_pred_init, x_size = inputs  # 修正後のデータ形式に合わせてアンパック
        if self.isCNN and self.fc_cnn_proj is not None:
            X = self.fc_cnn_proj(X)

        # Encoder
        for i in range(self.num_layers):
            z = self.layer_norms_enc1X[i](X)
            z = self.self_attentionsX[i](z, z)
            X = X + z

            z = self.layer_norms_enc2X[i](X)
            z = self.fcs_encX[i](z)
            X = X + z

        # Decoder
        # y_pred = tf.tile(self.set_emb, [tf.shape(X)[0], 1, 1]) # クラスタ中心なしバージョン
        y_pred = y_pred_init
        for i in range(self.num_layers):
            q = self.layer_norms_decq[i](y_pred)
            k = self.layer_norms_deck[i](X)

            q = self.cross_attentions[i](q, k)
            y_pred = y_pred + q

            q = self.layer_norms_dec2[i](y_pred)
            q = self.fcs_dec[i](q)
            y_pred = y_pred + q

        set_score = self.cross_set_score((X, y_pred))
        return set_score

    def cross_set_label(self, y):
        y = tf.reshape(y, [-1]) 
        y_rows = tf.tile(tf.expand_dims(y, -1), [1, tf.shape(y)[0]])
        y_cols = tf.tile(tf.expand_dims(y, 0), [tf.shape(y)[0], 1])
        labels = tf.cast(tf.equal(y_rows, y_cols), tf.float32)
        return labels

    def log_memory_usage(self, message):
        process = psutil.Process(os.getpid())
        # 必要に応じてメモリ使用量をログに記録

    def train_step(self, data):
        (X_batch, y_pred_init_batch, x_size), SetID = data  # 修正後のデータ形式に合わせてアンパック

        with tf.GradientTape() as tape:
            set_score = self((X_batch, y_pred_init_batch, x_size), training=True)
            y_true = self.cross_set_label(SetID)
            y_true = tf.linalg.set_diag(y_true, tf.zeros([tf.shape(y_true)[0]], dtype=tf.float32))
            loss = self.compiled_loss(y_true, set_score, regularization_losses=self.losses)

        # 勾配計算と適用
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # メトリックの更新
        self.compiled_metrics.update_state(y_true, set_score)

        # 結果の収集
        results = {"loss": loss}
        for metric in self.metrics:
            metric_value = metric.result()
            results[metric.name] = metric_value

        return results

    def test_step(self, data):
        (X_batch, y_pred_init_batch, x_size), SetID = data
        set_score = self((X_batch, y_pred_init_batch, x_size), training=False)
        y_true = self.cross_set_label(SetID)
        y_true = tf.linalg.set_diag(y_true, tf.zeros([tf.shape(y_true)[0]], dtype=tf.float32))
        loss = self.compiled_loss(y_true, set_score, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y_true, set_score)

        # 結果の収集
        results = {"loss": loss}
        for metric in self.metrics:
            metric_value = metric.result()
            results[metric.name] = metric_value

        return results