import json
from pathlib import Path
from PIL import Image
from typing import Dict


class DeepFurnitureDataset:
    """A simple dataset loader for DeepFurniture dataset."""
    
    def __init__(self, data_dir: str):
        """Initialize the dataset loader.
        
        Args:
            data_dir: Path to the dataset directory
        """
        self.data_dir = Path(data_dir)
        
        # Load metadata
        with open(self.data_dir / "metadata" / "categories.json") as f:
            self.categories = json.load(f)
        with open(self.data_dir / "metadata" / "styles.json") as f:
            self.styles = json.load(f)
        
        # Get scene directories
        self.scene_dirs = sorted(p for p in (self.data_dir / "scenes").iterdir() if p.is_dir())
    
    def __len__(self) -> int:
        """Get number of scenes in dataset."""
        return len(self.scene_dirs)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single scene with its annotations and related data.
        
        Returns:
            Dict containing:
            - scene_id (str): Scene identifier
            - image (PIL.Image): Scene image
            - depth (PIL.Image or None): Depth map if available
            - has_depth (bool): Whether depth map is available
            - instances (List[Dict]): List of furniture instances with:
                - category_id: Category identifier
                - category_name: Category name
                - identity_id: Furniture identity identifier
                - style_ids: List of style identifiers
                - style_names: List of style names
                - bounding_box: Dict with xmin, ymin, xmax, ymax
                - segmentation: Segmentation mask
            - furniture_previews (Dict[str, PIL.Image]): Mapping of furniture IDs to preview images
        """
        # Get scene directory
        scene_dir = self.scene_dirs[idx]
        scene_id = scene_dir.name
        
        # Load image
        image = Image.open(scene_dir / "image.jpg")
        
        # Load depth if available
        depth = None
        has_depth = False
        depth_path = scene_dir / "depth.png"
        if depth_path.exists():
            try:
                depth = Image.open(depth_path)
                has_depth = True
            except Exception as e:
                print(f"Warning: Failed to load depth map for scene {scene_id}: {e}")
        
        # Load annotation
        with open(scene_dir / "annotation.json") as f:
            annotation = json.load(f)
        
        # Process instances
        instances = []
        furniture_ids = set()
        for inst in annotation["instances"]:
            instance_data = {
                "category_id": inst["categoryID"],
                "category_name": inst["categoryName"],
                "identity_id": inst["identityID"],
                "style_ids": inst["styleIDs"],
                "style_names": inst["styleNames"],
                "bounding_box": {
                    "xmin": inst["boundingBox"]["xMin"],
                    "ymin": inst["boundingBox"]["yMin"],
                    "xmax": inst["boundingBox"]["xMax"],
                    "ymax": inst["boundingBox"]["yMax"],
                },
                "segmentation": inst["segmentation"] if "segmentation" in inst else None,
            }
            instances.append(instance_data)
            furniture_ids.add(str(inst["identityID"]))
        
        # Load furniture previews
        furniture_previews = {}
        for furniture_id in furniture_ids:
            furniture_path = self.data_dir / "furnitures" / f"{furniture_id}.jpg"
            if furniture_path.exists():
                try:
                    furniture_previews[furniture_id] = Image.open(furniture_path)
                except Exception as e:
                    print(f"Warning: Failed to load furniture preview {furniture_id}: {e}")
        
        return {
            "scene_id": scene_id,
            "image": image,
            "depth": depth,
            "has_depth": has_depth,
            "instances": instances,
            "furniture_previews": furniture_previews
        }
    
    def __iter__(self):
        """Iterate through all scenes."""
        for i in range(len(self)):
            yield self[i]