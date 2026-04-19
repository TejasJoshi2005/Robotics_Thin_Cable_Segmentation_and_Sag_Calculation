import os
import shutil
import random
from pathlib import Path

def split_dataset(image_dir, mask_dir, output_dir, split_ratio = 0.8, seed = 42):
    random.seed(seed)

    images_path = Path(image_dir)
    mask_path = Path(mask_dir)
    output_path = Path(output_dir)

    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path/ split / 'masks').mkdir(parents=True, exist_ok=True)


    image_files = sorted(list(images_path.rglob("*.jpg")))
    mask_files = sorted(list(mask_path.rglob("*.png")))

    if len(image_files) != len(mask_files):
        print("Images and masks mismatch")
        return
    
    paired_files = list(zip(image_files, mask_files))

    random.shuffle(paired_files)

    train_size = int(len(paired_files) * split_ratio)

    train_pairs = paired_files[:train_size]
    val_pairs = paired_files[train_size:]

    print(f"Total pairs: {len(paired_files)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}\n")

    def copy_files(pairs, split_name):
        for img_path, mask_path in pairs:

            dest_img = output_path / split_name / 'images' / img_path.name
            dest_mask = output_path / split_name / 'masks' / mask_path.name

            shutil.copy2(img_path, dest_img)
            shutil.copy2(mask_path, dest_mask)

    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')

split_dataset(image_dir="dataset", mask_dir="masks", output_dir="Split_Dataset")
