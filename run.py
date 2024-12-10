# run.py 内

import os
import sys
import argparse
import pickle
import numpy as np
import tensorflow as tf
import gzip

import util
import models
import make_datasets as data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

np.random.seed(42)
tf.random.set_seed(42)

def load_category_centers(center_path='uncompressed_data/category_centers.pkl.gz'):
    if not os.path.exists(center_path):
        print("category_centers.pkl.gz not found. seed_vectors=0")
        return None, 0
    with gzip.open(center_path, 'rb') as f:
        category_centers = pickle.load(f)
    rep_vec_num = category_centers.shape[0]
    return category_centers, rep_vec_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SMN model for set retrieval.')
    # 追加引数をparserに追加
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--train_category', action='store_true', help='Flag to train category model (現状は使用しません)')
    parser.add_argument('--train_set_matching', action='store_true', help='Flag to train set matching model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--max_item_num', type=int, default=5, help='Max item num per set')
    parser.add_argument('--data_dir', type=str, default='uncompressed_data', help='Directory containing splitted data')
    parser.add_argument('--baseMlp', type=int, default=512, help='Base MLP channel size')  # 追加
    parser.add_argument('--is_set_norm', type=int, default=0, help='Enable set normalization (0 or 1)')
    parser.add_argument('--is_cross_norm', type=int, default=0, help='Enable cross normalization (0 or 1)')
    parser.add_argument('--pretrained_mlp', type=int, default=0, help='Enable pretrained MLP (0 or 1)')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of set attention layers')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--mode', type=str, default='setRepVec_pivot', help='Mode of operation')
    parser.add_argument('--calc_set_sim', type=str, default='CS', help='Set similarity calculation method')
    parser.add_argument('--baseChn', type=int, default=32, help='Base channel size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--use_Cvec', type=int, default=1, help='Use category vectors (0 or 1)')
    parser.add_argument('--is_Cvec_linear', type=int, default=0, help='Use linear transformation for category vectors (0 or 1)')
    args = parser.parse_args()

    # 出力ディレクトリ作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'model'))
        os.makedirs(os.path.join(args.output_dir, 'result'))

    main_checkpoint_path = os.path.join(args.output_dir, "model/cp.weights.h5")
    main_result_path = os.path.join(args.output_dir, "result/result.pkl")

    # カテゴリ中心ベクトルをロードしてseed_vectorsとして利用
    seed_vectors, rep_vec_num = load_category_centers('uncompressed_data/category_centers.pkl.gz')
    args.rep_vec_num = rep_vec_num

    # データジェネレータ用意
    train_generator = data.trainDataGenerator(data_dir=args.data_dir, batch_size=args.batch_size, max_item_num=args.max_item_num, mode='train')
    x_valid, x_size_valid, y_valid = train_generator.data_generation_val()

    if args.train_category:
        print("Note: train_category flag is ON (not explicitly training category model here)")
    if args.train_set_matching:
        print("Note: train_set_matching flag is ON, we will train SMN model for set retrieval.")

    # SMNモデル構築時にseed_initとしてcategory_centersを渡す
    model = models.SMN(
        isCNN=False,
        is_set_norm=(args.is_set_norm == 1),
        is_cross_norm=(args.is_cross_norm == 1),
        is_TrainableMLP=(args.pretrained_mlp == 1),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mode=args.mode,
        calc_set_sim=args.calc_set_sim,
        baseChn=args.baseChn,
        baseMlp=args.baseMlp,
        rep_vec_num=args.rep_vec_num,
        seed_init=seed_vectors,  # カテゴリ中心ベクトルを初期シードとして利用
        use_Cvec=(args.use_Cvec == 1),
        is_Cvec_linear=(args.is_Cvec_linear == 1),
        max_item_num=args.max_item_num  # SMNにmax_item_numを渡す
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=util.Set_hinge_loss,
                  metrics=[util.Set_accuracy],
                  run_eagerly=True)

    if args.train_set_matching:
        # コールバック
        cp_callback = tf.keras.callbacks.ModelCheckpoint(main_checkpoint_path, monitor='val_Set_accuracy', save_weights_only=True, mode='max', save_best_only=True, verbose=1)
        cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_Set_accuracy', patience=args.patience, mode='max', min_delta=0.001, verbose=1)

        history = model.fit(
            train_generator,
            epochs=args.epochs,
            validation_data=((x_valid, x_size_valid), y_valid),
            shuffle=True,
            callbacks=[cp_callback, cp_earlystopping]
        )

        # validation
        loss, accuracy = model.evaluate((x_valid, x_size_valid), y_valid)
        print(f"Validation loss: {loss}, Validation accuracy: {accuracy}")

        acc = history.history['Set_accuracy']
        val_acc = history.history['val_Set_accuracy']
        loss_hist = history.history['loss']
        val_loss_hist = history.history['val_loss']

        util.plotLossACC(args.output_dir, loss_hist, val_loss_hist, acc, val_acc)

        with open(main_result_path, 'wb') as fp:
            pickle.dump(acc, fp)
            pickle.dump(val_acc, fp)
            pickle.dump(loss_hist, fp)
            pickle.dump(val_loss_hist, fp)
    else:
        # モデルパラメータ読み込み
        if os.path.exists(main_checkpoint_path):
            model.load_weights(main_checkpoint_path)
            print("Loaded model weights.")
        else:
            print("No trained model found. Exiting.")

    # モデルパラメータの最終保存
    model.save_weights(main_checkpoint_path)
    print("Done.")
