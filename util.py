# util.py
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import pdb

def parser_run():
    parser = argparse.ArgumentParser(description='Set Retrieval Training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=10, help='Patience for EarlyStopping')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    return parser

def plotLossACC(path, batch_size, loss, val_loss, acc, val_acc, recall_at_5=None, val_recall_at_5=None, mrr=None, val_mrr=None):
    epochs = np.arange(len(acc))
    if recall_at_5 is not None and mrr is not None:
        fig = plt.figure(figsize=(18, 5))  # サイズを拡大
    else:
        fig = plt.figure(figsize=(12, 5))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    
    # Accuracy Plot
    ax1 = fig.add_subplot(1, 2, 1 if recall_at_5 is None else 1)
    ax1.plot(epochs, acc, 'bo-', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy')  # 色を区別
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    if recall_at_5 is not None and mrr is not None:
        # Loss Plot
        ax2 = fig.add_subplot(1, 3, 2)
    else:
        # Loss Plot
        ax2 = fig.add_subplot(1, 2, 2)
    
    ax2.plot(epochs, loss, 'bo-', label='Training Loss')
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss')  # 色を区別
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, max(max(loss), max(val_loss)) + 1)  # 損失の最大値に基づいて設定
    ax2.legend()
    
    if recall_at_5 is not None and mrr is not None:
        # Recall@5 and MRR Plot
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(epochs, recall_at_5, 'bo-', label='Training Recall@5')
        ax3.plot(epochs, val_recall_at_5, 'r-', label='Validation Recall@5')
        ax3.plot(epochs, mrr, 'go-', label='Training MRR')
        ax3.plot(epochs, val_mrr, 'm-', label='Validation MRR')
        ax3.set_title('Recall@5 and MRR')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Metrics')
        ax3.legend()
        ax3.set_ylim(0, 1)
    
    if recall_at_5 is not None and mrr is not None:
        plt.savefig(os.path.join(path, f"loss_acc_recall_mrr_{batch_size}.png"))
    else:
        plt.savefig(os.path.join(path, f"loss_acc_{batch_size}.png"))
    plt.close()

def Set_hinge_loss(y_true, y_pred):
    margin = 1.0
    # 正例: y_true == 1
    pos_loss = y_true * tf.square(tf.maximum(0., margin - y_pred))
    # 負例: y_true == 0
    neg_loss = (1 - y_true) * tf.square(tf.maximum(0., y_pred))
    loss = tf.reduce_mean(pos_loss + neg_loss)
    return loss

def mask_diagonal(sim_matrix, mask_value=-1e9):
    """
    類似度行列の対角成分をマスク値に設定します。
    
    Args:
        sim_matrix: Tensor of shape (batch_size, batch_size)
        mask_value: float, マスクに使用する値
    
    Returns:
        Tensor of shape (batch_size, batch_size) with diagonal masked
    """
    batch_size = tf.shape(sim_matrix)[0]
    # 対角マスクを作成
    mask = tf.linalg.diag(tf.ones(batch_size))
    # 対角成分をmask_valueに設定
    masked_sim_matrix = tf.where(tf.equal(mask, 1), tf.ones_like(sim_matrix) * mask_value, sim_matrix)
    return masked_sim_matrix

class SetAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='Set_accuracy', **kwargs):
        super(SetAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true, y_pred: (batch_size, batch_size)
        
        # 対角成分をマスク
        y_pred_masked = mask_diagonal(y_pred)
        
        # 最も類似度が高いサンプルを選択
        top1_pred = tf.cast(tf.argmax(y_pred_masked, axis=1), tf.int32)
        
        # y_trueは同カテゴリサンプルの存在を示す二値行列
        # 各サンプルのトップ1予測が同カテゴリに属するかをチェック
        # y_trueは既に対角成分が0に設定されていると仮定
        
        # 正解ラベルとの一致を確認
        # 各サンプルに対して、予測されたインデックスが同カテゴリに属するか
        matches = tf.gather_nd(
            y_true, 
            tf.stack([
                tf.range(tf.shape(y_true)[0], dtype=tf.int32), 
                top1_pred
            ], axis=1)
        )
        matches = tf.cast(matches, self.dtype)

        if sample_weight is not None:
            matches = tf.multiply(matches, sample_weight)

        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(matches), self.dtype))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        self.correct.assign(0.)
        self.total.assign(0.)

class SetRecallAtK(tf.keras.metrics.Metric):
    def __init__(self, k=5, name='Set_Recall@K', **kwargs):
        super(SetRecallAtK, self).__init__(name=name, **kwargs)
        self.k = k
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: (batch_size, batch_size)
        # y_pred: (batch_size, batch_size)
        
        # 対角成分をマスク
        y_pred_masked = mask_diagonal(y_pred, mask_value=-1e9)
        
        # Top K predictions
        top_k = tf.nn.top_k(y_pred_masked, k=self.k).indices  # shape: (batch_size, k)
        
        # y_trueをtop_kに基づいて取得
        sorted_y_true = tf.gather(y_true, top_k, batch_dims=1)  # shape: (batch_size, k)
        
        # 各サンプルに対して、Top K内に正解が存在するかを確認
        tp = tf.reduce_any(tf.equal(sorted_y_true, 1), axis=1)  # shape: (batch_size,)
        tp = tf.cast(tp, self.dtype)
        
        if sample_weight is not None:
            tp = tf.multiply(tp, sample_weight)
        
        self.true_positives.assign_add(tf.reduce_sum(tp))
        self.total.assign_add(tf.cast(tf.size(tp), self.dtype))
    
    def result(self):
        return tf.math.divide_no_nan(self.true_positives, self.total)
    
    def reset_states(self):
        self.true_positives.assign(0.)
        self.total.assign(0.)

class MeanReciprocalRank(tf.keras.metrics.Metric):
    def __init__(self, name='MRR', **kwargs):
        super(MeanReciprocalRank, self).__init__(name=name, **kwargs)
        self.reciprocal_rank = self.add_weight(name='rr', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: (batch_size, batch_size)
        # y_pred: (batch_size, batch_size)
        
        # 対角成分をマスク
        y_pred_masked = mask_diagonal(y_pred, mask_value=-1e9)
        
        # ソートされたインデックスを取得（降順）
        sorted_indices = tf.argsort(y_pred_masked, direction='DESCENDING', axis=1)
        
        # y_trueをソートされたインデックスに基づいて取得
        sorted_y_true = tf.gather(y_true, sorted_indices, batch_dims=1)  # shape: (batch_size, k)
        
        # 各サンプルで最初に出現する1のインデックス（順位）
        first_rank = tf.argmax(sorted_y_true, axis=1) + 1  # ranks start at 1
        
        # 正解が存在しない場合の処理
        has_positive = tf.reduce_any(y_true, axis=1)
        first_rank = tf.where(has_positive, first_rank, tf.ones_like(first_rank) * tf.shape(y_true)[1])
        
        reciprocal = 1.0 / tf.cast(first_rank, tf.float32)
        
        if sample_weight is not None:
            reciprocal = reciprocal * sample_weight
        
        self.reciprocal_rank.assign_add(tf.reduce_sum(reciprocal))
        self.count.assign_add(tf.cast(tf.size(reciprocal), self.dtype))
    
    def result(self):
        return tf.math.divide_no_nan(self.reciprocal_rank, self.count)
    
    def reset_states(self):
        self.reciprocal_rank.assign(0.)
        self.count.assign(0.)
