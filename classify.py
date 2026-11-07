import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# ==============================================================================
# 1. Model & Transformation Initialization
# ==============================================================================
print("Loading MobileNetV2 pretrained model and ImageNet weights...")
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# Set model to evaluation mode (turns off dropout, locks batch norm)
model.eval()

# Retrieve the exact preprocessing transforms bundled with the weights
preprocess = weights.transforms()

# Retrieve the 1,000 ImageNet category names
categories = weights.meta["categories"]
print("Model initialized successfully.\n")


# ==============================================================================
# 2. Single-Image Classification Function
# ==============================================================================
def classify_image(image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Runs a single image through the MobileNetV2 inference pipeline:

    1. Loads and standardizes image to 3-channel RGB.
    2. Applies bundled transforms (resize, crop, normalization).
    3. Adds a batch dimension [1, 3, 224, 224].
    4. Computes forward pass with no gradient tracking.
    5. Applies Softmax to output top-k predictions with confidence scores.
    """
    img = Image.open(image_path).convert("RGB")

    # Preprocess image and add batch dimension [1, C, H, W]
    batch = preprocess(img).unsqueeze(0)

    # Disable gradient tracking for inference performance
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits[0], dim=0)

    # Retrieve top-k probabilities and category indices
    top = torch.topk(probs, top_k)
    return [(categories[idx], float(prob)) for prob, idx in zip(top.values, top.indices)]


# ==============================================================================
# 3. Batch Directory Processing & Visualization
# ==============================================================================
def process_images_directory(dir_path: str = "images") -> None:
    """Iterates through image files, executes classification with error handling,

    prints top-3 predictions, and saves a composite results visualization.
    """
    directory = Path(dir_path)

    if not directory.exists() or not directory.is_dir():
        print(f"Directory '{dir_path}' does not exist! Please create it and add photos.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in valid_extensions]

    if not image_files:
        print(f"No valid image files found in '{dir_path}'. Add some .jpg or .png photos to test!")
        return

    print(f"Found {len(image_files)} image(s) in '{dir_path}'. Starting inference...\n")
    print("=" * 60)

    successful_results = []

    for img_file in image_files:
        print(f"Image: {img_file.name}")
        try:
            predictions = classify_image(str(img_file), top_k=3)
            for rank, (label, score) in enumerate(predictions, start=1):
                print(f"  Top-{rank}: {label:<25} ({score * 100:.2f}%)")
            print("-" * 60)
            successful_results.append((img_file, predictions[0][0], predictions[0][1]))
        except (UnidentifiedImageError, OSError) as e:
            print(f"  [Error] Skipping corrupt or unreadable image file: {e}")
            print("-" * 60)
        except Exception as e:
            print(f"  [Unexpected Error] Could not process {img_file.name}: {e}")
            print("-" * 60)

    # Generate visual summary plot if any images succeeded
    if successful_results:
        n_imgs = len(successful_results)
        cols = min(4, n_imgs)
        rows = (n_imgs + cols - 1) // cols
        plt.figure(figsize=(4 * cols, 4 * rows))

        for idx, (path, top_label, conf) in enumerate(successful_results):
            plt.subplot(rows, cols, idx + 1)
            raw_img = Image.open(path).convert("RGB")
            plt.imshow(raw_img)
            plt.title(f"{top_label}\n({conf * 100:.1f}%)", fontsize=10)
            plt.axis("off")

        plt.tight_layout()
        output_plot = "results.png"
        plt.savefig(output_plot, dpi=150)
        plt.close()
        print(f"\nVisual inference summary saved to '{output_plot}'.")


if __name__ == "__main__":
    process_images_directory("images")