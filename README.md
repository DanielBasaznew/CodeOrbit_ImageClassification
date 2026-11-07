# CodeOrbit Image Classification Pipeline (Day 4)

An end-to-end image classification pipeline in PyTorch using a pretrained **MobileNetV2** model trained on ImageNet (1.2 million images, 1,000 object categories).

## How It Works
1. **Input Standardization:** Reads image files and applies PIL `.convert("RGB")` so grayscale, PNG RGBA (transparency), and color images all map uniformly to standard 3-channel RGB.
2. **Preprocessing Transforms:** Applies `MobileNet_V2_Weights.DEFAULT.transforms()` to resize and center-crop to 224x224 pixels, convert to a floating-point PyTorch tensor, and normalize pixel values against ImageNet mean and standard deviation.
3. **Inference Execution:** Evaluates input in `model.eval()` mode with `torch.no_grad()` to lock batch normalization/dropout and disable autograd graph tracking for optimal inference speed.
4. **Softmax Probabilities:** Normalizes output logits with `torch.softmax()` so class scores form a proper probability distribution summing to 1.0 (100%), and extracts the Top-3 predictions using `torch.topk()`.

## How to Run It

1. Activate your virtual environment:
   ```bash
   source .venv/Scripts/activate  # On Windows Git Bash