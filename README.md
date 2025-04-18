# DeepFurniture: End-to-End Furniture Set Retrieval (Transformer-based)

This repository extends the original DeepFurniture dataset with a fully working experimental pipeline for **furniture set retrieval**, using a Transformer-inspired network with optional cycle-consistency and a novel contrastive loss approach.

Below you will find:

1. **Original DeepFurniture dataset information**
2. **Instructions on training and testing our Transformer-based retrieval model**
3. **Details on how to generate collages of retrieval results, compute global rank logs, and interpret training logs**

---

## Dataset Information

This dataset is introduced in the paper: [Furnishing Your Room by What You See: An End-to-End Furniture Set Retrieval Framework with Rich Annotated Benchmark Dataset](https://arxiv.org/abs/1911.09299)

Project: [https://www.kujiale.com/festatic/furnitureSetRetrieval](https://www.kujiale.com/festatic/furnitureSetRetrieval)

Dataset license: [AFL-3.0](https://opensource.org/licenses/AFL-3.0)

A large-scale dataset for furniture understanding, featuring **photo-realistic rendered indoor scenes** with **high-quality 3D furniture models**. The dataset contains about 24k indoor images, 170k furniture instances, and 20k unique furniture identities, all rendered by the leading industry-level rendering engines in [COOHOM](https://coohom.com).

### Key Features

- **Photo-Realistic Rendering**: All indoor scenes are rendered using professional rendering engines, providing realistic lighting, shadows, and textures.
- **High-Quality 3D Models**: Each furniture identity is derived from a professional 3D model, ensuring accurate geometry and material representation.
- **Rich Annotations**: Hierarchical annotations at image, instance, and identity levels.

### Dataset Overview

DeepFurniture provides hierarchical annotations at three levels:

- **Image Level**: Photo-realistic scenes, each with a scene category and optional depth map.
- **Instance Level**: Bounding boxes and per-pixel segmentation masks for furniture instances in each scene.
- **Identity Level**: High-quality 3D previews of each furniture model.

#### Statistics

- ~24,000 scenes
- ~170,000 furniture instances
- ~20,000 unique furniture identities
- 11 furniture categories
- 11 style tags

### Dataset Structure

The dataset is split for distribution efficiency:

```
data/
├── scenes/           # Rendered indoor images
├── furnitures/       # 3D furniture model preview images
├── queries/          # Cropped query instances
└── metadata/         # JSON/JSONL metadata and definitions
```

### Using the Dataset

1. **Download & Extraction**

   ```bash
   git lfs install  # Ensure Git LFS is installed
   git clone https://huggingface.co/datasets/byliu/DeepFurniture
   ```

   Optionally, uncompress via:

   ```bash
   cd DeepFurniture
   bash uncompress_dataset.sh -s data -t uncompressed_data
   ```

2. **Data Format**

   - **Scene Data**: Scenes in JPG format, optional depth PNG, JSON annotations.
   - **Furniture Data**: Preview images of each furniture model + JSONL metadata.
   - **Queries**: Cropped furniture patches from scenes.

3. **Loading Sample Scenes**

   ```python
   from deepfurniture import DeepFurnitureDataset

   dataset = DeepFurnitureDataset("uncompressed_data")
   scene = dataset[0]
   print(f"Scene ID: {scene['scene_id']}")
   print(f"Number of instances: {len(scene['instances'])}")
   ```

4. **Scene Visualization**

   ```bash
   python visualize_html.py --dataset ./uncompressed_data --scene_idx 101 --output scene_101.html
   ```

### Acknowledgments

- Created by [COOHOM](https://coohom.com)/[酷家乐](https://kujiale.com)
- Rendered with professional interior design engines
- Thanks to the community of 3D artists and designers

---

## Preprocessing Pipeline

The following scripts convert the raw DeepFurniture dataset into a form suitable for Transformer-based furniture set retrieval.

### 1. `preprocess_fix.py`: Scene-Level Feature Extraction

```bash
python preprocess_fix.py \
  --image_dir ./uncompressed_data/furnitures/ \
  --categories_json_path ./uncompressed_data/metadata/categories.json \
  --furnitures_jsonl_path ./uncompressed_data/metadata/furnitures.jsonl \
  --annotations_json_path ./uncompressed_data/annotations.json \
  --raw_data_path ./uncompressed_data/raw_data_Fix.pkl \
  --epochs 10 \
  --batch_size 32
```

**Key Steps**

1. **Image Preprocessing**: Resize to 224×224 and apply VGG16 preprocessing.
2. **Metadata Loading**: Load categories (`categories.json`), furniture-level mappings (`furnitures.jsonl`), and scene annotations (`annotations.json`).
3. **Category Model Training**: A classification model based on `VGG16 (include_top=True)` is trained to predict furniture categories.
4. **Feature Extraction**: Extract 256D features from the trained model, aggregated by scene.
5. **Output Files**:
   - `raw_data_Fix.pkl`: Dictionary mapping scene IDs to item-wise features, indices, and category IDs.
   - `category_centers_Fix.pkl.gz`: Mean feature vectors per category from training data.

```python
{
  'L3D122I5X725QUPFR7OQFQ3P3X2888': {
    'features': np.ndarray,  # shape (N_items, 256)
    'item_indices': List[int],
    'category_ids': np.ndarray  # shape (N_items,)
  },
  ...
}
```

### 2. `make_datasets_Norm.py`: Train/Val/Test Splitting

```bash
python make_datasets_Norm.py \
  --raw_data_path ./uncompressed_data/raw_data_Fix.pkl \
  --output_dir ./datasets/ \
  --test_size 0.2 \
  --validation_size 0.1 \
  --min_items 6 \
  --max_items 16 \
  --max_item_num 8
```

**Behavior**

- Filters scenes with at least `min_items` items.
- Randomly splits items in a scene into:
  - Query set (up to `max_item_num // 2` items)
  - Positive set (remaining items)
- Applies padding/truncation to ensure fixed size.
- Outputs:
  - `train_Fix.pkl`
  - `validation_Fix.pkl`
  - `test_Fix.pkl`

Each record in these `.pkl` files includes:

```python
{
  'scene_id': str,
  'queries': np.ndarray,       # (num_query_items, 256)
  'positives': np.ndarray,     # (num_pos_items, 256)
  'query_category_ids': np.ndarray,
  'positive_category_ids': np.ndarray
}
```

Use these `.pkl` files with `run_0211.py` for training/testing.

---

## Transformer-Based Retrieval Experiment

This repository adds an **end-to-end training and evaluation pipeline** for set retrieval using a Transformer-inspired network with cycle-consistency and a combined contrastive loss (`CLCatNeg+CLPPNeg`). The main scripts are:

```
DeepFurniture/
├── run_0211.py            # Entry point for training/test
├── models_0211.py         # Main retrieval model (SetRetrievalModel)
├── data_generator_0211.py # Loads .pkl dataset
├── util_0211.py           # Additional utilities (plotting, ranking, collages)
├── datasets/              # train_Fix.pkl, val_Fix.pkl, test_Fix.pkl
├── uncompressed_data/     # Scenes and furniture images for collage
└── output/                # Model weights, logs, result CSV
```

### 1. Environment Setup

```bash
conda create -n deepfurniture python=3.10
conda activate deepfurniture
pip install tensorflow numpy matplotlib pandas scikit-learn pillow psutil
```

### 2. Prepare `.pkl` Data Files

- Place `train_Fix.pkl`, `validation_Fix.pkl`, and `test_Fix.pkl` in `datasets/`.
- (Optional) If you want collage visualization, also ensure `uncompressed_data/` is unzipped.

### 3. Training

Example:

```bash
python run_0211.py \
  --mode train \
  --embedding_dim 256 \
  --num_layers 2 \
  --num_heads 2 \
  --batch_size 32 \
  --epochs 10 \
  --patience 5 \
  --output_dir output/my_experiment \
  --use_CLNeg_loss
```

### 4. Testing

```bash
python run_0211.py \
  --mode test \
  --final_weights output/my_experiment/best_model.weights.h5 \
  --embedding_dim 256 \
  --num_layers 2 \
  --num_heads 2 \
  --batch_size 32 \
  --use_CLNeg_loss
```

### 5. Logs and Visualization

- Accuracy & loss plots: `output/my_experiment/loss_acc_32.png`
- Global rank logs: appended to `output/my_experiment/result.csv`
- Visual collages (retrieval results): `collages/`

### 6. Example Workflow

```bash
# Train
python run_0211.py --mode train --batch_size 32 --epochs 5 --output_dir output/debug

# Test
python run_0211.py --mode test --batch_size 32 \
  --final_weights output/debug/best_model.weights.h5

# Visual results saved to:
#   - output/debug/result.csv
#   - collages/
```

---

## Citation

```bibtex
@article{liu2019furnishing,
  title={Furnishing Your Room by What You See: An End-to-End Furniture Set Retrieval Framework with Rich Annotated Benchmark Dataset},
  author={Bingyuan Liu and Jiantao Zhang and Xiaoting Zhang and Wei Zhang and Chuanhui Yu and Yuan Zhou},
  journal={arXiv preprint arXiv:1911.09299},
  year={2019},
}
```

