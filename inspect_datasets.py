import pickle

def inspect_dataset(file_path):
    with open(file_path, 'rb') as f:
        X_Q, X_P, Y = pickle.load(f)
        print(f"{file_path}:")
        print(f"  X_Q shape: {X_Q.shape}")  # 例: (N, 8, 128)
        print(f"  X_P shape: {X_P.shape}")  # 例: (N, 8, 128)
        print(f"  Y shape: {Y.shape}")      # 例: (N,)
        print(f"  Example Y: {Y[:5]}")

if __name__ == "__main__":
    inspect_dataset('/data1/yamazono/setRetrieval/DeepFurniture/uncompressed_data/datasets/train.pkl')
    inspect_dataset('/data1/yamazono/setRetrieval/DeepFurniture/uncompressed_data/datasets/validation.pkl')
    inspect_dataset('/data1/yamazono/setRetrieval/DeepFurniture/uncompressed_data/datasets/test.pkl')
