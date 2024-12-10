import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import glob

def read_jsonl(file_path):
    """
    JSONLファイルを読み込み、リスト形式で返す。
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e} in line: {line}")
    return data

def plot_category_distribution(df, category_mapping, output_path=None):
    """
    カテゴリの分布を棒グラフでプロット。
    """
    # カテゴリIDをカテゴリ名にマッピング
    df['category_label'] = df['category_id'].astype(str).map(category_mapping)
    df = df.dropna(subset=['category_label'])  # マッピングされなかったカテゴリを除外

    plt.figure(figsize=(14, 7))
    sns.countplot(
        x='category_label',
        data=df,
        palette='viridis',
        hue='category_label',      # hueをxと同じに設定
        dodge=False,               # hueが設定された場合、棒が重ならないようにする
        legend=False               # 凡例を非表示にする
    )
    plt.title('Category Distribution', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_style_distribution(style_counts, style_mapping, output_path=None):
    """
    スタイルの分布を棒グラフでプロット。
    """
    # スタイルIDが0の場合は「No Style」と表示しないため除外
    # 既存のスタイルIDをラベルとして設定
    style_labels = {}
    for style_id in sorted(style_counts.keys()):
        # スタイルIDが0の場合はスキップ（No Styleを除外）
        if style_id == 0:
            continue
        # スタイルIDを文字列としてマッピング辞書を参照
        style_labels[style_id] = style_mapping.get(str(style_id), f'Style {style_id}')

    # データフレーム作成
    style_df = pd.DataFrame(list(style_counts.items()), columns=['style_id', 'count'])
    # スタイルIDが0の場合を除外
    style_df = style_df[style_df['style_id'] != 0]
    style_df['style_label'] = style_df['style_id'].apply(lambda x: style_labels.get(x, f'Style {x}'))

    plt.figure(figsize=(14, 7))
    sns.barplot(
        x='style_label',
        y='count',
        data=style_df,
        palette='magma',
        hue='style_label',         # hueをxと同じに設定
        dodge=False,               # hueが設定された場合、棒が重ならないようにする
        legend=False               # 凡例を非表示にする
    )
    plt.title('Style Distribution', fontsize=16)
    plt.xlabel('Style', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_items_per_scene_distribution(items_per_scene, output_path=None):
    """
    シーンごとのアイテム数の分布を棒グラフでプロット。
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(items_per_scene, bins=range(1, max(items_per_scene)+2), kde=False, color='skyblue', edgecolor='black')
    plt.title('Number of Items per Scene', fontsize=16)
    plt.xlabel('Number of Items', fontsize=14)
    plt.ylabel('Number of Scenes', fontsize=14)
    plt.xticks(range(1, max(items_per_scene)+1))
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    plt.show()

def count_items_per_scene(scenes_dir):
    """
    各シーンに含まれるアイテム数をカウントし、リストで返す。
    """
    items_per_scene = []
    annotation_files = glob.glob(os.path.join(scenes_dir, '*', 'annotation.json'))
    total_scenes = len(annotation_files)
    print(f"Found {total_scenes} annotation.json files.")

    for idx, annotation_file in enumerate(annotation_files, 1):
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                instances = data.get('instances', [])
                items_count = len(instances)
                items_per_scene.append(items_count)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in file {annotation_file}: {e}")
        except Exception as e:
            print(f"Error processing file {annotation_file}: {e}")

        if idx % 1000 == 0:
            print(f"Processed {idx} / {total_scenes} files.")

    return items_per_scene

def main():
    # ファイルパスの設定
    jsonl_file = 'uncompressed_data/metadata/furnitures.jsonl'  # 実際のファイルパスに置き換えてください
    scenes_dir = 'uncompressed_data/scenes'  # シーンディレクトリのパス

    # カテゴリIDとカテゴリ名のマッピング
    category_mapping = {
        "1": "cabinet#shelf",
        "2": "table",
        "3": "chair#stool",
        "4": "lamp",
        "5": "door",
        "6": "bed",
        "7": "sofa",
        "8": "plant",
        "9": "decoration",
        "10": "curtain",
        "11": "home-appliance"
    }

    # スタイルIDとスタイル名のマッピング
    style_mapping = {
        "1": "modern",
        "2": "country",
        "3": "European#American",
        "4": "Chinese",
        "5": "Japanese",
        "6": "Mediterranean",
        "7": "Southeast-Asian",
        "8": "Nordic",
        "9": "Industrial",
        "10": "electric",          # "electic" を "electric" に修正
        "11": "other"
    }

    # JSONLファイルの読み込み
    data = read_jsonl(jsonl_file)
    print(f"Total items in JSONL file: {len(data)}")

    # データフレームの作成
    df = pd.DataFrame(data)

    # スタイルなし（style_idsがNone）のアイテムを除外
    initial_count = len(df)
    df = df[df['style_ids'].notnull()]
    excluded_count = initial_count - len(df)
    if excluded_count > 0:
        print(f"Excluded {excluded_count} items with No Style.")

    # カテゴリ分布のプロット
    plot_category_distribution(df, category_mapping, output_path='category_distribution.png')

    # スタイルIDのカウント（複数スタイルを持つ場合は全てカウント）
    all_style_ids = []
    for styles in df['style_ids']:
        if styles is not None:
            all_style_ids.extend(styles)
        # else: スタイルなしを除外するため何もしない

    style_counts = Counter(all_style_ids)
    print("Style Counts:")
    print(style_counts)

    # スタイル分布のプロット
    plot_style_distribution(style_counts, style_mapping, output_path='style_distribution.png')

    # 各シーンに含まれるアイテム数のカウント
    items_per_scene = count_items_per_scene(scenes_dir)
    print(f"Total scenes processed: {len(items_per_scene)}")
    print(f"Items per scene: {Counter(items_per_scene)}")

    # シーンごとのアイテム数分布のプロット
    plot_items_per_scene_distribution(items_per_scene, output_path='items_per_scene_distribution.png')

if __name__ == '__main__':
    main()
