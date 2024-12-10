import os
import json
import numpy as np
from tqdm import tqdm
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pdb
import argparse
import gzip

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_categories(categories_json_path):
    with open(categories_json_path, 'r') as f:
        categories = json.load(f)
    return categories

def load_furniture_labels(furnitures_jsonl_path):
    furniture_labels = {}
    with open(furnitures_jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            furniture_id = data['furniture_id']
            category_id = data['category_id']
            furniture_labels[furniture_id] = {
                'category_id': category_id
            }
    return furniture_labels

def load_image_paths_and_labels(image_dir, furniture_labels):
    image_paths = []
    category_labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            furniture_id = os.path.splitext(filename)[0]
            if furniture_id in furniture_labels:
                image_paths.append(os.path.join(image_dir, filename))
                category_labels.append(furniture_labels[furniture_id]['category_id'])
    return image_paths, category_labels

def preprocess_image(image_path, target_size=(224, 224)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def prepare_data(image_paths, labels):
    images = []
    for path in tqdm(image_paths, desc='Loading images'):
        image = preprocess_image(path)
        images.append(image)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def build_simple_category_model(num_classes):
    inputs = tf.keras.Input(shape=(224,224,3))
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='feature_layer_cat')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='category_output')(x)
    model = tf.keras.models.Model(inputs, outputs)
    for layer in base_model.layers:
        layer.trainable = False
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='uncompressed_data/furnitures', help='Directory containing furniture images')
    parser.add_argument('--categories_json_path', type=str, default='uncompressed_data/metadata/categories.json', help='Path to categories.json')
    parser.add_argument('--furnitures_jsonl_path', type=str, default='uncompressed_data/metadata/furnitures.jsonl', help='Path to furnitures.jsonl')
    parser.add_argument('--raw_data_path', type=str, default='uncompressed_data/raw_data.pkl', help='Path to save raw data')
    parser.add_argument('--category_centers_path', type=str, default='uncompressed_data/category_centers.pkl.gz', help='Path to save category centers')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training (デモ用に2エポック)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    data_dir = args.image_dir
    categories_json_path = args.categories_json_path
    furnitures_jsonl_path = args.furnitures_jsonl_path
    raw_data_path = args.raw_data_path
    category_centers_path = args.category_centers_path

    # メタ情報読み込み
    categories = load_categories(categories_json_path)
    furniture_labels = load_furniture_labels(furnitures_jsonl_path)

    # 画像パスとラベル取得
    image_paths, category_labels = load_image_paths_and_labels(data_dir, furniture_labels)

    # カテゴリID→インデックス変換
    category_label_set = sorted(list(set(category_labels)))
    category_label_to_index = {lbl: i for i,lbl in enumerate(category_label_set)}
    category_labels = np.array([category_label_to_index[lbl] for lbl in category_labels])
    num_category_classes = len(category_label_set)

    # 画像読み込み
    images, category_labels = prepare_data(image_paths, category_labels)

    # カテゴリモデル学習
    category_model = build_simple_category_model(num_category_classes)
    category_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
    category_model.fit(images, category_labels, epochs=args.epochs, batch_size=args.batch_size) # デモ用に2epoch程度

    # 特徴抽出
    cat_feature_model = tf.keras.Model(inputs=category_model.input, outputs=category_model.get_layer('feature_layer_cat').output)
    cat_features = cat_feature_model.predict(images, batch_size=args.batch_size)

    # カテゴリのクラスタ中心を計算
    category_centers = compute_category_centers(cat_features, category_labels, num_category_classes)

    # クラスタ中心を保存
    with gzip.open(category_centers_path, 'wb') as f:
        pickle.dump(category_centers, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Category centers have been saved as {category_centers_path}')

    # raw_data.pkl に保存
    raw_data = {
        'features': cat_features,  # 128次元
        'category_labels': category_labels
    }

    if not os.path.exists(os.path.dirname(raw_data_path)):
        os.makedirs(os.path.dirname(raw_data_path))

    with open(raw_data_path,'wb') as f:
        pickle.dump(raw_data,f)

    print(f"raw_data.pkl has been created successfully and stored in '{raw_data_path}' directory.")

def compute_category_centers(features, labels, num_classes):
    centers = np.zeros((num_classes, features.shape[1]))
    for i in range(num_classes):
        class_features = features[labels == i]
        if class_features.shape[0] > 0:
            centers[i] = class_features.mean(axis=0)
        else:
            centers[i] = 0
    return centers

if __name__ == '__main__':
    main()
