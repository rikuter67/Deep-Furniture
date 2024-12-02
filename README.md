---
license: afl-3.0
task_categories:
- object-detection
- image-segmentation
size_categories:
- 10K<n<100K
---
# DeepFurniture Dataset (created by [COOHOM](https://coohom.com)/[酷家乐](https://kujiale.com))

This dataset is introduced in our paper:
[Furnishing Your Room by What You See: An End-to-End Furniture Set Retrieval Framework with Rich Annotated Benchmark Dataset](https://arxiv.org/abs/1911.09299)

Project: https://www.kujiale.com/festatic/furnitureSetRetrieval

<img src="visualizations/overview.png" width="100%"/>

A large-scale dataset for furniture understanding, featuring **photo-realistic rendered indoor scenes** with **high-quality 3D furniture models**. The dataset contains about 24k indoor images, 170k furniture instances, and 20k unique furniture identities, all rendered by the leading industry-level rendering engines in [COOHOM](https://coohom.com).


## Key Features

- **Photo-Realistic Rendering**: All indoor scenes are rendered using professional rendering engines, providing realistic lighting, shadows, and textures
- **High-Quality 3D Models**: Each furniture identity is derived from a professional 3D model, ensuring accurate geometry and material representation
- **Rich Annotations**: Hierarchical annotations at image, instance, and identity levels


## Dataset Overview

DeepFurniture provides hierarchical annotations at three levels:
- **Image Level**: Professional rendered indoor scenes with scene category and depth map
- **Instance Level**: Bounding boxes and per-pixel masks for furniture instances in scenes
- **Identity Level**: High-quality rendered previews of 3D furniture models.

### Statistics
- Total scenes: ~24,000 photo-realistic rendered images
- Total furniture instances: ~170,000 annotated instances in scenes
- Unique furniture identities: ~20,000 3D models with preview renderings
- Categories: 11 furniture types
- Style tags: 11 different styles

## Benchmarks

This dataset supports three main benchmarks:
1. Furniture Detection/Segmentation
2. Furniture Instance Retrieval 
3. Furniture Retrieval

For benchmark details and baselines, please refer to our paper.

## Dataset Structure

The dataset is organized in chunks for efficient distribution:

```
data/
├── scenes/           # Photo-realistic rendered indoor scenes
├── furnitures/       # High-quality 3D model preview renders
├── queries/          # Query instance images from scenes
└── metadata/         # Dataset information and indices
    ├── categories.json       # Furniture category definitions
    ├── styles.json          # Style tag definitions
    ├── dataset_info.json    # Dataset statistics and information
    ├── furnitures.jsonl     # Furniture metadata
    └── *_index.json        # Chunk index files
```

## Using the Dataset

### 1. Download and Extraction

```bash
# Clone the repository
git lfs install  # Make sure Git LFS is installed
git clone https://huggingface.co/datasets/byliu/DeepFurniture
```

[optional] Uncompress the dataset by the provided script.

Note: the current dataset loader is only available for uncompressed assets. So, if you want to use the provided dataset loader, you'll need to uncompress the assets firstly.
The dataset loader for compressed assets is TBD.
```
cd DeepFurniture
bash uncompress_dataset.sh -s data -t uncompressed_data
```

### 2. Data Format

#### Scene Data
- **Image**: RGB images in JPG format
- **Depth**: Depth maps in PNG format
- **Annotation**: JSON files containing:
  ```json
  {
    "instances": [
      {
        "numberID": 1,
        "boundingBox": {
          "xMin": int,
          "xMax": int,
          "yMin": int,
          "yMax": int
        },
        "styleIDs": [int],
        "styleNames": [str],
        "segmentation": [int], # COCO format RLE encoding
        "identityID": int,
        "categoryID": int,
        "categoryName": str
      }
    ]
  }
  ```

#### Furniture Data
- Preview images of 3D models in JPG format
- Metadata in JSONL format containing category and style information

#### Query Data
- Cropped furniture instances from scenes
- Filename format: `[furnitureID]_[instanceIndex]_[sceneID].jpg`

### 3. Loading the Dataset

```python
from deepfurniture import DeepFurnitureDataset

# Initialize dataset
dataset = DeepFurnitureDataset("path/to/uncompressed_data")

# Access a scene
scene = dataset[0]
print(f"Scene ID: {scene['scene_id']}")
print(f"Number of instances: {len(scene['instances'])}")

# Access furniture instances
for instance in scene['instances']:
    print(f"Category: {instance['category_name']}")
    print(f"Style(s): {instance['style_names']}")
```

### 4. To visualize each indoor scene
```
python visualize_html.py --dataset ./uncompressed_data --scene_idx 101 --output scene_101.html
```


## Acknowledgments

- Dataset created and owned by [COOHOM](https://coohom.com)/[酷家乐](https://kujiale.com)
- Rendered using the leading interior design platform in [COOHOM](https://coohom.com)/[酷家乐](https://kujiale.com)
- Thanks to millions of designers and artists who contributed to the 3D models and designs

If you use this dataset, please cite:
```bibtex
@article{liu2019furnishing,
  title={Furnishing Your Room by What You See: An End-to-End Furniture Set Retrieval Framework with Rich Annotated Benchmark Dataset},
  author={Bingyuan Liu and Jiantao Zhang and Xiaoting Zhang and Wei Zhang and Chuanhui Yu and Yuan Zhou},
  journal={arXiv preprint arXiv:1911.09299},
  year={2019},
}
```