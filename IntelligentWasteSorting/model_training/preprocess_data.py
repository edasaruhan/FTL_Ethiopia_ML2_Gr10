import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Define paths
DATASET_DIR = Path("../datasets")
TRASHNET_DIR = DATASET_DIR / "trashnet/data/dataset-resized"
KAGGLE_DIR = DATASET_DIR / "DATASET"  # Adjust based on Kaggle dataset structure
OUTPUT_DIR = DATASET_DIR / "preprocessed"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

# Categories
CATEGORIES = ["paper", "plastic", "metal", "glass", "cardboard", "trash"]

# Create output directories
for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    for category in CATEGORIES:
        (dir_path / category).mkdir(parents=True, exist_ok=True)

def preprocess_image(image_path, output_path, size=(224, 224)):
    """Preprocess image: resize and normalize."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return False
    img = cv2.resize(img, size)
    img = img / 255.0  # Normalize to [0, 1]
    cv2.imwrite(str(output_path), img * 255)  # Save as image
    return True

def process_dataset(input_dir, dataset_name):
    """Process images from a dataset and split into train/val/test."""
    image_paths = []
    labels = []
    
    for category in CATEGORIES:
        category_dir = input_dir / category
        if not category_dir.exists():
            print(f"Category {category} not found in {dataset_name}")
            continue
        for img_path in category_dir.glob("*.jpg"):
            image_paths.append(img_path)
            labels.append(category)
    
    # Split dataset: 70% train, 15% val, 15% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Process and save images
    for split, paths, split_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels)
    ]:
        split_dir = OUTPUT_DIR / split
        for img_path, label in zip(paths, split_labels):
            output_path = split_dir / label / img_path.name
            preprocess_image(img_path, output_path)

def main():
    print("Processing TrashNet dataset...")
    process_dataset(TRASHNET_DIR, "TrashNet")
    print("Processing Kaggle dataset...")
    process_dataset(KAGGLE_DIR, "Kaggle")
    print("Preprocessing complete. Data saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()