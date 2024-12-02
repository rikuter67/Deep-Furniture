import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from deepfurniture import DeepFurnitureDataset
from pycocotools import mask as mask_utils

def save_image_base64(image):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode()


def create_instance_visualization(scene_data):
    """Create combined instance visualization with both masks and bboxes."""
    image = scene_data['image']
    instances = scene_data['instances']
    
    # Image dimensions for boundary checking
    img_width, img_height = image.size
    
    # Start with image at half opacity
    vis_img = np.array(image, dtype=np.float32) * 0.5
    
    # Get all segmentations
    segmentations = []
    for inst in instances:
        if inst['segmentation']:
            rle = {
                'counts': inst['segmentation'],
                'size': [img_height, img_width]
            }
            segmentations.append(rle)
    
    # Create color map for instances with distinct colors
    colors = np.array([
        [0.9, 0.1, 0.1],  # Red
        [0.1, 0.9, 0.1],  # Green
        [0.1, 0.1, 0.9],  # Blue
        [0.9, 0.9, 0.1],  # Yellow
        [0.9, 0.1, 0.9],  # Magenta
        [0.1, 0.9, 0.9],  # Cyan
        [0.9, 0.5, 0.1],  # Orange
        [0.5, 0.9, 0.1],  # Lime
        [0.5, 0.1, 0.9],  # Purple
    ])
    colors = np.tile(colors, (len(instances) // len(colors) + 1, 1))[:len(instances)]
    
    # Draw instance masks with higher opacity
    if segmentations:
        if isinstance(segmentations[0]['counts'], (list, tuple)):
            segmentations = mask_utils.frPyObjects(
                segmentations, img_height, img_width
            )
        masks = mask_utils.decode(segmentations)
        
        for idx in range(masks.shape[2]):
            color = colors[idx]
            mask = masks[:, :, idx]
            for c in range(3):
                vis_img[:, :, c] += mask * np.array(image)[:, :, c] * 0.7 * color[c]
    
    # Convert to PIL for drawing bounding boxes
    vis_img = Image.fromarray(np.uint8(np.clip(vis_img, 0, 255)))
    draw = ImageDraw.Draw(vis_img)
    
    # Try to load a font for better text rendering
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
    
    # Constants for text and box drawing
    text_padding = 4
    text_height = 24
    text_width = 200
    corner_length = 20
    
    # Draw bounding boxes with labels
    for idx, (instance, color) in enumerate(zip(instances, colors)):
        bbox = instance['bounding_box']
        color_tuple = tuple(int(c * 255) for c in color)
        
        # Calculate label
        furniture_id = instance['identity_id']
        category = instance['category_name']
        label = f"{category} ({furniture_id})"
        
        # Draw bbox with double lines for better visibility
        for offset in [2, 1]:
            draw.rectangle([
                max(0, bbox['xmin'] - offset),
                max(0, bbox['ymin'] - offset),
                min(img_width - 1, bbox['xmax'] + offset),
                min(img_height - 1, bbox['ymax'] + offset)
            ], outline=color_tuple, width=2)
        
        # Determine text position (handle boundary cases)
        # First try above the bbox
        text_y = bbox['ymin'] - text_height - text_padding
        if text_y < 0:  # If no space above, try below
            text_y = bbox['ymax'] + text_padding
            
        # Handle x position
        text_x = bbox['xmin']
        # If text would go beyond right edge, align to right edge
        if text_x + text_width > img_width:
            text_x = max(0, img_width - text_width)
        
        # Draw background for text
        text_pos = (text_x, text_y)
        draw.rectangle([
            text_pos[0] - 2,
            text_pos[1] - 2,
            min(img_width - 1, text_pos[0] + text_width),
            min(img_height - 1, text_pos[1] + text_height)
        ], fill='black')
        
        # Draw text
        draw.text(text_pos, label, fill=color_tuple, font=font)
        
        # Add corner markers with boundary checking
        corners = [
            (bbox['xmin'], bbox['ymin']),  # Top-left
            (bbox['xmax'], bbox['ymin']),  # Top-right
            (bbox['xmin'], bbox['ymax']),  # Bottom-left
            (bbox['xmax'], bbox['ymax'])   # Bottom-right
        ]
        
        for x, y in corners:
            # Ensure corner markers stay within image bounds
            # Horizontal lines
            x1 = max(0, x - corner_length)
            x2 = min(img_width - 1, x + corner_length)
            draw.line([(x1, y), (x2, y)], fill=color_tuple, width=3)
            
            # Vertical lines
            y1 = max(0, y - corner_length)
            y2 = min(img_height - 1, y + corner_length)
            draw.line([(x, y1), (x, y2)], fill=color_tuple, width=3)
    
    return vis_img


def process_depth_map(depth_image):
    """Process depth map for better visualization.
    
    Args:
        depth_image: PIL Image of depth map
    Returns:
        Processed depth map as PIL Image
    """
    # Convert to numpy array
    depth = np.array(depth_image)
    
    # Normalize depth to 0-1 range
    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    # Apply colormap (viridis-like)
    colored_depth = np.zeros((*depth.shape, 3))
    colored_depth[..., 0] = (1 - depth) * 0.4  # Red channel
    colored_depth[..., 1] = np.abs(depth - 0.5) * 0.8  # Green channel
    colored_depth[..., 2] = depth * 0.8  # Blue channel
    
    # Convert to uint8 and then to PIL
    colored_depth = (colored_depth * 255).astype(np.uint8)
    return Image.fromarray(colored_depth)


def visualize_html(dataset, scene_idx, output_path='scene.html'):
    """Generate HTML visualization for a scene."""
    scene_data = dataset[scene_idx]
    
    # Create visualizations
    instance_vis = create_instance_visualization(scene_data)
    
    depth_vis = None
    if scene_data['depth']:
        depth_vis = process_depth_map(scene_data['depth'])
    # Get base64 encoded images
    scene_img = save_image_base64(scene_data['image'])
    instance_vis = save_image_base64(instance_vis)
    depth_img = save_image_base64(depth_vis) if depth_vis else None
    
    # Create HTML with minimal CSS
    html = f'''
    <html>
    <head>
        <style>
            body {{ font-family: Arial; margin: 20px; max-width: 2000px; margin: 0 auto; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
            .main-images {{ 
                grid-template-columns: repeat(auto-fit, minmax(800px, 1fr)); 
                margin: 20px 0;
            }}
            .card {{ 
                border: 1px solid #ddd; 
                padding: 15px; 
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .card h3 {{ 
                font-size: 20px;
                margin-bottom: 15px;
            }}
            img {{ max-width: 100%; height: auto; }}
            h1 {{ 
                color: #333; 
                font-size: 32px;
                text-align: center;
                margin: 30px 0;
            }}
            h2 {{ 
                color: #333; 
                font-size: 28px;
                margin: 25px 0;
            }}
            .instance-info {{ 
                color: #444; 
                font-size: 16px;
                line-height: 1.4;
            }}
            .main-images img {{
                width: 100%;
                object-fit: contain;
                max-height: 800px;  /* Increased max height */
            }}
        </style>
    </head>
    <body>
        <h1>Scene ID: {scene_data['scene_id']}</h1>
        
        <h2>Scene Visualizations</h2>
        <div class="grid main-images">
            <div class="card">
                <h3>Original Scene</h3>
                <img src="data:image/png;base64,{scene_img}">
            </div>
            <div class="card">
                <h3>Instance Visualization (Masks + Bboxes)</h3>
                <img src="data:image/png;base64,{instance_vis}">
            </div>
            {f'<div class="card"><h3>Depth Map</h3><img src="data:image/png;base64,{depth_img}"></div>' if depth_img else ''}
        </div>
        
        <h2>Furniture Instances</h2>
        <div class="grid">
    '''
    
    # Add furniture previews
    for instance in scene_data['instances']:
        furniture_id = str(instance['identity_id'])
        if furniture_id in scene_data['furniture_previews']:
            preview = save_image_base64(scene_data['furniture_previews'][furniture_id])
            bbox = instance['bounding_box']
            
            html += f'''
            <div class="card">
                <h3>{instance['category_name']} (ID: {furniture_id})</h3>
                <div class="instance-info">
                    <p>Style: {', '.join(instance['style_names'])}</p>
                    <p>BBox: ({bbox['xmin']}, {bbox['ymin']}, {bbox['xmax']}, {bbox['ymax']})</p>
                </div>
                <img src="data:image/png;base64,{preview}">
            </div>
            '''
    
    html += '''
        </div>
    </body>
    </html>
    '''
    
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Visualization saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene_idx', type=int, required=True)
    parser.add_argument('--output', default='scene.html')
    args = parser.parse_args()

    dataset = DeepFurnitureDataset(args.dataset)
    visualize_html(dataset, args.scene_idx, args.output)
