import json
import cv2
import numpy as np
import os
from pycocotools import mask as maskUtils


dataset_dir = r'C:\Users\tejas\Desktop\Robotics Project\CableDrivenRobotCableModel\train'

json_path = os.path.join(dataset_dir, '_annotations.coco.json')
output_mask_dir = os.path.join(dataset_dir, 'masks')

os.makedirs(output_mask_dir, exist_ok=True)

print("Reading coco json")
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Map image IDs to file names and sizes
images_info = {img['id']: img for img in coco_data['images']}

print(f"Found {len(images_info)} images, generating masks using pycocotools")

# Draw the masks
for img_id, img_info in images_info.items():
    # Create a blank black image (0)
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    
    # Get all annotations for this specific image
    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    
    for ann in anns:
        seg = ann['segmentation']
        
        if isinstance(seg, list):
            for poly_coords in seg:
                poly = np.array(poly_coords, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [poly], 255)
                
        elif isinstance(seg, dict):
            # pycocotools requires the string to be encoded as bytes
            if isinstance(seg['counts'], str):
                seg['counts'] = seg['counts'].encode('utf-8')
            
            # Decode returns a binary mask (array of 0s and 1s)
            rle_mask = maskUtils.decode(seg)
            
            # Multiply by 255 to turn the 1s into bright white, and merge with the main mask
            mask = np.maximum(mask, rle_mask * 255)
                
    # Save the generated mask
    base_name = os.path.splitext(img_info['file_name'])[0]
    mask_filename = f"{base_name}_mask.png"
    cv2.imwrite(os.path.join(output_mask_dir, mask_filename), mask)

print(f"Generated {len(images_info)} perfect PNG masks in:\n{output_mask_dir}")