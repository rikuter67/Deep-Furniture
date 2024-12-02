from deepfurniture import DeepFurnitureDataset

# Initialize dataset
dataset = DeepFurnitureDataset("uncompressed_data")

# Access a scene
scene = dataset[*] # Get the *th scene
print(f"Scene ID: {scene['scene_id']}")
print(f"Number of instances: {len(scene['instances'])}")

# Access furniture instances
for instance in scene['instances']:
    print(f"Category: {instance['category_name']}")
    print(f"Style(s): {instance['style_names']}")