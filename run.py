import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def load_categories(categories_json_path):
    """カテゴリーIDと名前のマッピングを読み込む"""
    with open(categories_json_path, 'r') as f:
        categories = json.load(f)
    return categories

def load_furniture_labels(furnitures_jsonl_path):
    """家具IDとカテゴリーIDのマッピングを作成"""
    furniture_labels = {}
    with open(furnitures_jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            furniture_id = data['furniture_id']
            category_id = data['category_id']
            furniture_labels[furniture_id] = category_id
    return furniture_labels

def load_image_paths_and_labels(image_dir, furniture_labels):
    """画像パスと対応するラベルのリストを作成"""
    image_paths = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            furniture_id = os.path.splitext(filename)[0]
            if furniture_id in furniture_labels:
                image_paths.append(os.path.join(image_dir, filename))
                labels.append(furniture_labels[furniture_id])
    return image_paths, labels

def preprocess_image(image_path, target_size=(224, 224)):
    """画像を読み込み、前処理を行う"""
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    # VGG16用の前処理を適用
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def prepare_data(image_paths, labels):
    """画像とラベルを読み込み、NumPy配列として返す"""
    images = []
    for path in tqdm(image_paths, desc='Loading images'):
        image = preprocess_image(path)
        images.append(image)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def build_model(num_classes):
    """VGG16をベースとしたモデルを構築"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    
    # 4096 -> 2048 -> 512 -> 128 の3層のMLPを追加
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='feature_layer')(x)
    
    # 最終的な分類層
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 畳み込み層を凍結
    for layer in base_model.layers:
        layer.trainable = False

    return model

def main():
    # データのパスを設定
    data_dir = 'uncompressed_data'
    categories_json_path = os.path.join(data_dir, 'metadata', 'categories.json')
    furnitures_jsonl_path = os.path.join(data_dir, 'metadata', 'furnitures.jsonl')
    image_dir = os.path.join(data_dir, 'furnitures')

    # カテゴリーと家具のラベルを読み込む
    categories = load_categories(categories_json_path)
    furniture_labels = load_furniture_labels(furnitures_jsonl_path)

    # 画像パスとラベルを取得
    image_paths, labels = load_image_paths_and_labels(image_dir, furniture_labels)

    # ラベルを整数のインデックスに変換
    label_set = sorted(list(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(label_set)}
    labels = [label_to_index[label] for label in labels]

    num_classes = len(label_set)
    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {len(labels)}')

    # データを読み込み、前処理
    images, labels = prepare_data(image_paths, labels)

    # データを訓練用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels)

    # モデルを構築
    model = build_model(num_classes)

    # モデルをコンパイル
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # コールバックの設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # モデルの訓練
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, checkpoint])

    # モデルの評価
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy: {test_acc:.4f}')

    # 特徴量の抽出（訓練データとテストデータを合わせて）
    feature_model = Model(inputs=model.input, outputs=model.get_layer('feature_layer').output)
    features = feature_model.predict(images)

    # 特徴量とラベルをpickleで保存
    with open('features.pkl', 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Features and labels have been saved as features.pkl')

if __name__ == '__main__':
    main()
