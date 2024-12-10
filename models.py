import tensorflow as tf
import numpy as np
import pdb

class layer_normalization(tf.keras.layers.Layer):
    def __init__(self, size_d, epsilon=1e-3, is_set_norm=False):
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.is_set_norm = is_set_norm

    def call(self, x, x_size):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        norm = (x - mean)/(std + self.epsilon)
        return norm

class set_attention(tf.keras.layers.Layer):
    def __init__(self, head_size=64, num_heads=2):
        super(set_attention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.linearQ = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)
        self.linearK = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)
        self.linearV = tf.keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=False)
        self.linearH = tf.keras.layers.Dense(units=self.head_size, use_bias=False)

    def call(self, q, k):
        q_ = self.linearQ(q)
        k_ = self.linearK(k)
        v_ = self.linearV(k)
        sqrt_head_size = tf.sqrt(tf.cast(self.head_size, tf.float32))

        shape_q = tf.shape(q_)
        shape_k = tf.shape(k_)
        nSet = shape_q[0]
        nItemMax_q = shape_q[1]
        nItemMax_k = shape_k[1]

        q_ = tf.reshape(q_, [nSet, nItemMax_q, self.num_heads, self.head_size])
        k_ = tf.reshape(k_, [nSet, nItemMax_k, self.num_heads, self.head_size])
        v_ = tf.reshape(v_, [nSet, nItemMax_k, self.num_heads, self.head_size])

        # Transpose for multi-head attention
        q_ = tf.transpose(q_, [0, 2, 1, 3])  # (nSet, num_heads, nItemMax_q, head_size)
        k_ = tf.transpose(k_, [0, 2, 1, 3])  # (nSet, num_heads, nItemMax_k, head_size)
        v_ = tf.transpose(v_, [0, 2, 1, 3])  # (nSet, num_heads, nItemMax_k, head_size)

        # Compute attention scores
        score = tf.matmul(q_, k_, transpose_b=True) / sqrt_head_size  # (nSet, num_heads, nItemMax_q, nItemMax_k)
        score = tf.nn.softmax(score, axis=-1)  # Softmax over the last axis

        # Weighted sum of values
        weighted_v = tf.matmul(score, v_)  # (nSet, num_heads, nItemMax_q, head_size)

        # Reshape back to (nSet, nItemMax_q, num_heads * head_size)
        weighted_v = tf.transpose(weighted_v, [0, 2, 1, 3])
        weighted_v = tf.reshape(weighted_v, [nSet, nItemMax_q, self.num_heads * self.head_size])

        # Final linear layer
        output = self.linearH(weighted_v)  # (nSet, nItemMax_q, head_size)
        return output
# models.py 内

class cross_set_score(tf.keras.layers.Layer):
    def __init__(self):
        super(cross_set_score, self).__init__()
        # 必要に応じて追加のレイヤーをここに定義

    def call(self, x):
        gallery, query = x  # gallery と query は共に (batch_size, 1, 64) の形状

        # 形状を (batch_size, 64) に変更
        gallery = tf.squeeze(gallery, axis=1)  # (batch_size, 64)
        query = tf.squeeze(query, axis=1)      # (batch_size, 64)

        # セット間の類似度を計算（ドット積）
        similarity = tf.matmul(gallery, query, transpose_b=True)  # (batch_size, batch_size)

        return similarity  # 出力形状は (batch_size, batch_size)

# models.py 内

class SMN(tf.keras.Model):
    def __init__(self, 
                 isCNN=False,
                 is_set_norm=False, 
                 is_cross_norm=False, 
                 is_TrainableMLP=False,
                 num_layers=1, 
                 num_heads=2, 
                 mode='setRepVec_pivot', 
                 calc_set_sim='CS', 
                 baseChn=32, 
                 baseMlp=512, 
                 rep_vec_num=41, 
                 seed_init=None, 
                 use_Cvec=True,
                 is_Cvec_linear=False,
                 max_item_num=5):
        super(SMN, self).__init__()
        self.isCNN = isCNN
        self.is_set_norm = is_set_norm
        self.is_TrainableMLP = is_TrainableMLP
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mode = mode
        self.calc_set_sim = calc_set_sim
        self.rep_vec_num = rep_vec_num
        self.use_Cvec = use_Cvec
        self.is_Cvec_linear = is_Cvec_linear
        self.max_item_num = max_item_num  # メンバー変数として保持
        self.baseMlp = baseMlp  # 追加

        self.dim = 64
        self.fc_item = tf.keras.layers.Dense(self.dim, use_bias=False)

        self.set_att = [set_attention(head_size=self.dim, num_heads=self.num_heads) for _ in range(self.num_layers)]
        self.ln_set = [layer_normalization(size_d=self.dim, is_set_norm=self.is_set_norm) for _ in range(self.num_layers)]
        self.cross_set_score = cross_set_score()  # 修正済み

        # Initialize seed vectors if provided
        if seed_init is not None:
            self.seed_init = tf.constant(seed_init, dtype=tf.float32)
        else:
            self.seed_init = 0

    def build(self, input_shape):
        super(SMN, self).build(input_shape)
        # 必要に応じて追加のビルド処理

    def cross_set_label(self, y):
        y_rows = tf.tile(tf.expand_dims(y, -1), [1, tf.shape(y)[0]])
        y_cols = tf.tile(tf.expand_dims(y, 0), [tf.shape(y)[0], 1])
        labels = tf.cast(y_rows == y_cols, tf.float32)
        return labels

    def call(self, inputs, training=False):
        x, x_size = inputs
        # x: (batch_size, max_item_num, feature_dim=256)
        x_proj = self.fc_item(x)  # (batch_size, max_item_num, dim=64)

        z = x_proj
        for i in range(self.num_layers):
            z_norm = self.ln_set[i](z, x_size)
            z_att = self.set_att[i](z_norm, z_norm)
            z = z + z_att

        # set representation
        set_repr = tf.reduce_mean(z, axis=1)  # (batch_size, dim=64)
        set_repr = tf.expand_dims(set_repr, axis=1)  # (batch_size, 1, dim=64)
        return set_repr

    def train_step(self, data):
        (x, x_size), y = data
        with tf.GradientTape() as tape:
            set_repr = self((x, x_size), training=True)  # (batch_size,1,64)
            y_true = self.cross_set_label(y)  # (batch_size, batch_size)
            y_true = tf.linalg.set_diag(y_true, tf.zeros([tf.shape(y_true)[0]], dtype=tf.float32))  # 自己との類似度を0に設定

            set_score = self.cross_set_score((set_repr, set_repr))  # (batch_size, batch_size)

            loss = self.compiled_loss(set_score, y_true, regularization_losses=self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(set_score, y_true)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (x, x_size), y = data
        set_repr = self((x, x_size), training=False)  # (batch_size,1,64)
        y_true = self.cross_set_label(y)  # (batch_size, batch_size)
        y_true = tf.linalg.set_diag(y_true, tf.zeros([tf.shape(y_true)[0]], dtype=tf.float32))  # 自己との類似度を0に設定

        set_score = self.cross_set_score((set_repr, set_repr))  # (batch_size, batch_size)
        self.compiled_loss(set_score, y_true, regularization_losses=self.losses)
        self.compiled_metrics.update_state(set_score, y_true)

        return {m.name: m.result() for m in self.metrics}
