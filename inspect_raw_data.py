import pickle

def inspect_raw_data(raw_data_path):
    with open(raw_data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # データ型の確認
    print(f"Type of raw_data: {type(raw_data)}")
    
    # データ内容の概要を表示
    if isinstance(raw_data, dict):
        print(f"Number of scenes: {len(raw_data)}")
        for i, (scene_id, data) in enumerate(raw_data.items()):
            print(f"Scene {i+1}: ID = {scene_id}")
            print(f"  Type of data: {type(data)}")
            if isinstance(data, dict):
                print(f"  Keys in data: {list(data.keys())}")
                if 'features' in data:
                    print(f"  Features shape: {data['features'].shape}")
                if 'item_indices' in data:
                    print(f"  Number of item indices: {len(data['item_indices'])}")
            if i >= 4:  # 最初の5シーンのみ表示
                break
    else:
        print("raw_data is not a dictionary. Please check the structure.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect raw_data.pkl")
    parser.add_argument('--raw_data_path', type=str, required=True, help='raw_data.pkl のパス')
    args = parser.parse_args()
    
    inspect_raw_data(args.raw_data_path)
