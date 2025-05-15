from ultralytics import YOLO
import numpy as np
import os
import torch
from glob import glob

def validate_labels(image_dir, label_dir):
    """
    Validates if each image has a corresponding label file and checks the label format.
    """
    print(f"\n🔍 Validating labels in: {image_dir} and {label_dir}")
    image_paths = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))
    missing_labels = []
    invalid_labels = []

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{img_name}.txt")

        if not os.path.exists(label_path):
            missing_labels.append(label_path)
            continue

        # Check YOLO format in label file
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_labels.append((label_path, line))
                    break
                try:
                    _ = [float(x) for x in parts[1:]]
                    if not all(0 <= float(x) <= 1 for x in parts[1:]):
                        invalid_labels.append((label_path, line))
                        break
                except ValueError:
                    invalid_labels.append((label_path, line))
                    break

    if missing_labels:
        print(f"❌ Missing labels: {len(missing_labels)} files")
    if invalid_labels:
        print(f"❌ Invalid format in: {len(invalid_labels)} files")
    if not missing_labels and not invalid_labels:
        print("✅ All labels validated successfully.\n")

    return not missing_labels and not invalid_labels


def train_yolo():
    # Validate labels before training
    base_path = "D:/gaura/Project/YOLO object detection/bone fracture detection/data"
    train_valid = validate_labels(
        image_dir=os.path.join(base_path, "train/images"),
        label_dir=os.path.join(base_path, "train/labels")
    )
    val_valid = validate_labels(
        image_dir=os.path.join(base_path, "test/images"),
        label_dir=os.path.join(base_path, "test/labels")
    )

    if not (train_valid and val_valid):
        print("❌ Label validation failed. Please fix the dataset before training.")
        return

    model = YOLO("yolov8n.pt")  # Load pre-trained model

    results = model.train(
        data="config.yaml",    # Path to the dataset configuration file
        epochs=20,             # Number of epochs to train
        device=0,              # Use GPU (device 0)
        save=True,             # Ensure saving is enabled
        save_period=1,         # Save weights after every epoch
        workers=4,             # Increase workers to improve data loading
        half=True,             # Enable mixed precision for faster training
    )

    # Save model manually with custom name
    weights_dir = "runs/detect/train/weights"
    best_path = os.path.join(weights_dir, "best.pt")
    last_path = os.path.join(weights_dir, "last.pt")

    if os.path.exists(best_path):
        torch.save(torch.load(best_path), "yolo_bonefracture_best.pt")
        print("✅ Saved: yolo_bonefracture_best.pt")
    elif os.path.exists(last_path):
        torch.save(torch.load(last_path), "yolo_bonefracture_last.pt")
        print("⚠️ Saved: yolo_bonefracture_last.pt (best.pt not available)")
    else:
        print("❌ No weights found to save manually.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    train_yolo()
