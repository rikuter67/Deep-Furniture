import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from util import parser_run, plotLossACC, Set_hinge_loss, SetAccuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau
from models import SMN
import gzip
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pdb

# Eager Execution を無効化（パフォーマンス向上のため）
tf.config.run_functions_eagerly(False)

# CUDA 関連の環境変数設定（必要に応じて）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU を選択
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

np.random.seed(42)
tf.random.set_seed(42)

def load_category_centers(center_path='category_centers.pkl.gz'):
    if not os.path.exists(center_path):
        print("category_centers.pkl.gz not found. seed_vectors=0")
        return None,0
    with gzip.open(center_path,'rb') as f:
        category_centers = pickle.load(f)
    rep_vec_num = category_centers.shape[0]
    return category_centers, rep_vec_num

def save_best_metrics(history, batch_size, output_dir):
    """最も良いvalidation accuracyの情報をCSVに保存"""
    # トレーニング履歴からデータを取得
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    acc = history.history.get('Set_accuracy', [])
    val_acc = history.history.get('val_Set_accuracy', [])

    best_idx = val_acc.index(max(val_acc))
    best_metrics = {'batch_size': batch_size, 'epoch': best_idx + 1, 'loss': loss[best_idx], 'val_loss': val_loss[best_idx], 'Set_accuracy': acc[best_idx], 'val_Set_accuracy': val_acc[best_idx]}
    # DataFrameに変換
    df = pd.DataFrame([best_metrics])

    # CSVに保存
    csv_path = os.path.join(output_dir, 'best_metrics.csv')
    df.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"Best metrics saved to {csv_path}")

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size=32, shuffle=True):
        self.data_path = data_path  # data_path を属性として保存
        with open(data_path, 'rb') as f:
            # データのロード
            # dataは (X_Q, X_P, Y, Y_cat_Q, Y_cat_P, x_sizes) の6つをロード
            self.X_Q, self.X_P, self.Y, self.Y_cat_Q, self.Y_cat_P, self.x_sizes = pickle.load(f)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.Y))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        # seed_vectorsをロード
        self.seed_vectors, self.rep_vec_num = load_category_centers('category_centers.pkl.gz')  # 例

    def __len__(self):
        return int(np.ceil(len(self.Y) / self.batch_size))

    def __getitem__(self, index):
        batch_inds = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_Q_batch = self.X_Q[batch_inds].astype(np.float32)  # (B,8,128)
        X_P_batch = self.X_P[batch_inds].astype(np.float32)  # (B,8,128)
        Y_cat_Q_batch = self.Y_cat_Q[batch_inds]  # (B,8)
        Y_cat_P_batch = self.Y_cat_P[batch_inds]  # (B,8)
        SetID = self.Y[batch_inds]  # (B,)

        # バッチ方向でconcat: (2B,8,128)
        X_batch = np.concatenate([X_Q_batch, X_P_batch], axis=0)  # (2B,8,128)
        SetID = np.concatenate([SetID, SetID], axis=0)  # (2B,)

        # x_sizeは一律で8とする（全て同じ長さと仮定）
        x_size = np.full(X_batch.shape[0], 8, dtype=np.float32)  # (2B,)

        # y_pred_init_Q の生成
        y_pred_init_Q = np.zeros((X_Q_batch.shape[0], X_Q_batch.shape[1], self.seed_vectors.shape[1]), dtype=np.float32)  # (B,8,128)
        mask_Q = Y_cat_Q_batch > 0  # (B,8)

        # カテゴリIDの範囲チェック
        if np.any(Y_cat_Q_batch > self.seed_vectors.shape[0]):
            raise ValueError(f"Y_cat_Q_batch contains invalid category IDs: {Y_cat_Q_batch[Y_cat_Q_batch > self.seed_vectors.shape[0]]}")
        if np.any(Y_cat_Q_batch < 0):
            raise ValueError(f"Y_cat_Q_batch contains negative category IDs: {Y_cat_Q_batch[Y_cat_Q_batch < 0]}")

        # 正しくインデックスを割り当て
        y_pred_init_Q[mask_Q] = self.seed_vectors[Y_cat_Q_batch[mask_Q] - 1]

        # y_pred_init_P の生成
        y_pred_init_P = np.zeros((X_P_batch.shape[0], X_P_batch.shape[1], self.seed_vectors.shape[1]), dtype=np.float32)  # (B,8,128)
        mask_P = Y_cat_P_batch > 0  # (B,8)

        # カテゴリIDの範囲チェック
        if np.any(Y_cat_P_batch > self.seed_vectors.shape[0]):
            raise ValueError(f"Y_cat_P_batch contains invalid category IDs: {Y_cat_P_batch[Y_cat_P_batch > self.seed_vectors.shape[0]]}")
        if np.any(Y_cat_P_batch < 0):
            raise ValueError(f"Y_cat_P_batch contains negative category IDs: {Y_cat_P_batch[Y_cat_P_batch < 0]}")

        y_pred_init_P[mask_P] = self.seed_vectors[Y_cat_P_batch[mask_P] - 1]

        # y_pred_init の結合: (2B,8,128)
        y_pred_init = np.concatenate([y_pred_init_P, y_pred_init_Q], axis=0)  # (2B,8,128)

        # モデルへの入力は ((X_batch, y_pred_init, x_size), SetID)
        return ((X_batch, y_pred_init, x_size), SetID)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def test_item_search(model, test_gen, output_dir, top_k=10):
    """
    テストデータでのアイテム検索を実行し、ランキング測度を計算する。

    Args:
        model: 学習済みモデル
        test_gen: テスト用データジェネレータ
        output_dir: 結果を保存するディレクトリ
        top_k: Top-k Accuracy を計算する際の k 値
    """
    # テストデータを全てロード
    print("Loading test data for item search...")
    all_queries = []
    all_positive_indices = []
    all_item_vectors = []
    all_set_ids = []
    
    # データジェネレータがDataGeneratorクラスのインスタンスなので、data_path属性にアクセス可能
    for batch in test_gen:
        (X_batch, y_pred_init, x_size), SetID = batch
        # X_batch: (2B,8,128)
        # y_pred_init: (2B,8,128)
        # SetID: (2B,)
        # 前半 B がクエリ、後半 B がポジティブアイテムと仮定
        B = X_batch.shape[0] // 2
        queries = X_batch[:B]  # (B,8,128)
        queries_pred_init = y_pred_init[:B]  # (B,8,128)
        positive_items = X_batch[B:]  # (B,8,128)
        positive_pred_init = y_pred_init[B:]  # (B,8,128)
        set_ids = SetID[:B]  # (B,)
        positive_set_ids = SetID[B:]  # (B,)
        
        all_queries.append((queries, queries_pred_init))
        all_positive_indices.extend(positive_set_ids.tolist())
        all_item_vectors.append((positive_items, positive_pred_init))
        all_set_ids.extend(set_ids.tolist())

        pdb.set_trace()

    # Concatenate all queries and items
    all_queries = np.concatenate([q[0] for q in all_queries], axis=0)  # (Total_B,8,128)
    all_queries_pred_init = np.concatenate([q[1] for q in all_queries], axis=0)  # (Total_B,8,128)
    all_item_vectors = np.concatenate([i[0] for i in all_item_vectors], axis=0)  # (Total_B,8,128)
    all_item_pred_init = np.concatenate([i[1] for i in all_item_vectors], axis=0)  # (Total_B,8,128)

    total_queries = all_queries.shape[0]
    total_items = all_item_vectors.shape[0]

    pdb.set_trace()

    print(f"Total queries: {total_queries}, Total items: {total_items}")

    # Create a new model that outputs set embeddings
    print("Modifying SMN to output set embeddings for test item search...")

    class SMN_Embedding(tf.keras.Model):
        def __init__(self, original_model, **kwargs):
            super(SMN_Embedding, self).__init__(**kwargs)
            self.original_model = original_model

        def call(self, inputs, training=False):
            X, y_pred_init, x_size = inputs
            if self.original_model.isCNN and self.original_model.fc_cnn_proj is not None:
                X = self.original_model.fc_cnn_proj(X)

            # Encoder
            for i in range(self.original_model.num_layers):
                z = self.original_model.layer_norms_enc1X[i](X)
                z = self.original_model.self_attentionsX[i](z, z)
                X = X + z

                z = self.original_model.layer_norms_enc2X[i](X)
                z = self.original_model.fcs_encX[i](z)
                X = X + z

            # Decoder
            y_pred = y_pred_init
            for i in range(self.original_model.num_layers):
                q = self.original_model.layer_norms_decq[i](y_pred)
                k = self.original_model.layer_norms_deck[i](X)

                q = self.original_model.cross_attentions[i](q, k)
                y_pred = y_pred + q

                q = self.original_model.layer_norms_dec2[i](y_pred)
                q = self.original_model.fcs_dec[i](q)
                y_pred = y_pred + q

            # Instead of cross_set_score, return y_pred as the embedding
            return y_pred  # (batch_size,8,128)

    # Instantiate the embedding model
    embedding_model = SMN_Embedding(model)
    embedding_model.compile(optimizer='adam')  # Dummy compile

    # Encode all items to obtain their embeddings
    print("Encoding all items to obtain embeddings...")
    item_embeddings = []
    # Define a new DataGenerator for items only
    class ItemGenerator(tf.keras.utils.Sequence):
        def __init__(self, item_vectors, item_pred_init, batch_size=128):
            self.item_vectors = item_vectors
            self.item_pred_init = item_pred_init
            self.batch_size = batch_size
            self.indexes = np.arange(len(self.item_vectors))

        def __len__(self):
            return int(np.ceil(len(self.item_vectors) / self.batch_size))

        def __getitem__(self, index):
            batch_inds = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            X_batch = self.item_vectors[batch_inds].astype(np.float32)
            y_pred_init = self.item_pred_init[batch_inds].astype(np.float32)
            x_size = np.full(X_batch.shape[0], 8, dtype=np.float32)
            return ((X_batch, y_pred_init, x_size), None)

        def on_epoch_end(self):
            pass

    item_gen = ItemGenerator(all_item_vectors, all_item_pred_init, batch_size=128)
    for batch in item_gen:
        (X_batch, y_pred_init, x_size), _ = batch
        embeddings = embedding_model((X_batch, y_pred_init, x_size), training=False)  # (B,8,128)
        embeddings = tf.reduce_mean(embeddings, axis=1)  # (B,128)
        item_embeddings.append(embeddings.numpy())
    item_embeddings = np.concatenate(item_embeddings, axis=0)  # (Total_Items,128)

    # Encode all queries to obtain their embeddings
    print("Encoding all queries to obtain embeddings...")
    query_embeddings = []
    class QueryGenerator(tf.keras.utils.Sequence):
        def __init__(self, query_vectors, query_pred_init, batch_size=128):
            self.query_vectors = query_vectors
            self.query_pred_init = query_pred_init
            self.batch_size = batch_size
            self.indexes = np.arange(len(self.query_vectors))

        def __len__(self):
            return int(np.ceil(len(self.query_vectors) / self.batch_size))

        def __getitem__(self, index):
            batch_inds = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            X_batch = self.query_vectors[batch_inds].astype(np.float32)
            y_pred_init = self.query_pred_init[batch_inds].astype(np.float32)
            x_size = np.full(X_batch.shape[0], 8, dtype=np.float32)
            return ((X_batch, y_pred_init, x_size), None)

        def on_epoch_end(self):
            pass

    query_gen = QueryGenerator(all_queries, all_queries_pred_init, batch_size=128)
    for batch in query_gen:
        (X_batch, y_pred_init, x_size), _ = batch
        embeddings = embedding_model((X_batch, y_pred_init, x_size), training=False)  # (B,8,128)
        embeddings = tf.reduce_mean(embeddings, axis=1)  # (B,128)
        query_embeddings.append(embeddings.numpy())
    query_embeddings = np.concatenate(query_embeddings, axis=0)  # (Total_Queries,128)

    # Compute cosine similarity between queries and items
    print("Computing cosine similarity between queries and items...")
    similarity_matrix = cosine_similarity(query_embeddings, item_embeddings)  # (Total_Queries, Total_Items)

    # For each query, find the rank of the positive item
    print("Calculating ranks of positive items...")
    ranks = []
    for i in range(total_queries):
        # Positive set ID
        positive_set_id = all_positive_indices[i]
        # Find the indices of items with the same set ID
        positive_indices = [idx for idx, set_id in enumerate(all_set_ids) if set_id == positive_set_id]
        if not positive_indices:
            print(f"No positive items found for SetID {positive_set_id} in query {i}. Skipping.")
            continue
        # Assuming the first occurrence is the correct positive item
        positive_index = positive_indices[0]
        
        # Similarity scores for this query
        sim_scores = similarity_matrix[i]
        # Rank the similarity scores in descending order
        ranked_indices = np.argsort(-sim_scores)
        
        # Find the rank of the positive item
        rank = np.where(ranked_indices == positive_index)[0][0] + 1  # Ranks start at 1
        ranks.append(rank)

    # Calculate evaluation metrics
    mrr = np.mean(1 / np.array(ranks))
    top_k_acc = np.mean(np.array(ranks) <= top_k)

    print(f"Test Results -> MRR: {mrr:.4f}, Top-{top_k} Accuracy: {top_k_acc:.4f}")

    # Save results to CSV
    result_df = pd.DataFrame({"Rank": ranks})
    csv_path = os.path.join(output_dir, "test_item_search_results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Test item search results saved to {csv_path}")



def main():
    # コマンドライン引数のパース
    parser = parser_run()
    args = parser.parse_args()

    # 出力ディレクトリの作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # データのパス
    train_path = os.path.join('uncompressed_data', 'datasets', 'train.pkl')
    val_path = os.path.join('uncompressed_data', 'datasets', 'validation.pkl')
    test_path = os.path.join('uncompressed_data', 'datasets', 'test.pkl')

    # データジェネレータの作成
    train_gen = DataGenerator(train_path, batch_size=args.batch_size, shuffle=True)
    val_gen = DataGenerator(val_path, batch_size=args.batch_size, shuffle=False)
    test_gen = DataGenerator(test_path, batch_size=args.batch_size, shuffle=False)

    # ReduceLROnPlateau コールバックの設定
    reduce_lr = ReduceLROnPlateau(monitor='val_Set_accuracy', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # モデルの初期化
    model = SMN(
        isCNN=False, 
        num_layers=1,
        num_heads=2,
        rep_vec_num=1,
        dim=128,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=Set_hinge_loss,
        metrics=[SetAccuracy()],  # カスタムメトリックを追加
        run_eagerly=True  # デバッグ目的で必要な場合のみ True
    )

    # コールバックの設定
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_Set_accuracy',  # メトリクス名を修正
        patience=args.patience,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.output_dir, 'best_model.weights.h5'),
        monitor='val_Set_accuracy',  # メトリクス名を修正
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    )

    # メトリクス名の確認
    print("Metrics names:", model.metrics_names)
    print("Registered metrics:", [metric.name for metric in model.metrics])

    # モデルのトレーニング
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=[early_stop, model_checkpoint, reduce_lr],
    )

    model.summary()

    # トレーニング履歴のプロット
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [0] * len(loss))
    acc = history.history.get('Set_accuracy', [0] * len(loss))
    val_acc = history.history.get('val_Set_accuracy', [0] * len(loss))

    save_best_metrics(history, args.batch_size, args.output_dir)

    # プロット
    plotLossACC(args.output_dir, args.batch_size, loss, val_loss, acc, val_acc)

    # テストデータでのアイテム検索
    test_item_search(model, test_gen, args.output_dir, top_k=10)

    # 最終モデルの保存
    model.save_weights(os.path.join(args.output_dir, 'final_model.weights.h5'))

    print("Training completed successfully.")

if __name__ == '__main__':
    main()
