# Engineering Journal — Day 4: Pretrained Image Classification

## 1. What I Studied
- **Pretrained Vision Architectures:** MobileNetV2 pre-trained on ImageNet (1.2 million images, 1,000 distinct classes).
- **Transformation Pipeline:** Raw image -> RGB conversion -> 224x224 resize/crop -> tensor conversion -> ImageNet mean/std normalization.
- **Inference Constraints:** Using `model.eval()` and `torch.no_grad()` to ensure deterministic output, zero out gradient graph overhead, and reduce RAM consumption.
- **Logit Calibration:** Transforming raw unconstrained logits into probabilities via `torch.softmax(dim=0)`.

## 2. Core Notes (Vision Models)
1. **Pretrained Weights:** MobileNetV2 uses pre-calculated filters capable of extracting hierarchical visual features (edges -> textures -> complex objects).
2. **Input Normalization:** Preprocessing must match the exact mean and standard deviation of ImageNet training data, or model predictions degrade.
3. **Batch Formatting:** Single images must be unsqueezed from `[3, 224, 224]` to `[1, 3, 224, 224]` to satisfy the PyTorch batch dimension requirement.
4. **Softmax Behavior:** Forces probabilities across all 1,000 categories to sum to 1.0 (100%).
5. **Probability Fragmentation:** In categories with extensive sub-classes (such as 120+ dog breeds), probabilities can split across near-identical classes, resulting in lower individual confidence values.

## 3. What I Built
- Configured `CodeOrbit_ImageClassification` with an isolated `.venv` using PyTorch CPU wheels.
- Developed `classify.py` containing:
  - Robust directory iteration with error handling for non-image or corrupt files.
  - Image preprocessing and unsqueezing.
  - Top-3 class extraction with formatted percentage readouts.
  - Automatic side-by-side plot generation exported to `results.png`.

## 4. Tests Run & Empirical Results
- **Armored Vehicle (`car.png`):** Predicted `jeep` (19.24%) and `half track` (7.16%). Correctly identified structural utility vehicle features.
- **Domestic Cat (`cat.png`):** Predicted `Egyptian cat` (34.56%) and `tabby` (18.05%). Strong identification of feline markings and face shapes.
- **Dog in Grass (`dog.png`):** Predicted `Staffordshire bullterrier` (3.71%) and `kelpie` (3.34%). Demonstrated probability diffusion over fine-grained dog breed categories.
- **Visualization:** Generated `results.png` displaying the three images with their Top-1 predictions and confidence percentages.

## 5. End-to-End Explanation (5 Lines)
1. `classify.py` iterates through the `images/` directory, ignoring non-image files.
2. `Image.open().convert("RGB")` standardizes every image to 3 channels.
3. `weights.transforms()` resizes, crops to 224x224, and normalizes pixel values.
4. A forward pass via `model(batch)` runs inside a `torch.no_grad()` context block.
5. `torch.softmax()` computes probability distributions, and `torch.topk()` logs the top-3 predictions and saves `results.png`.

## 6. What I Don't Understand Yet / Areas to Explore
- How to apply transfer learning by replacing MobileNetV2's final 1,000-class linear head with a custom classification layer (e.g., binary dog vs. cat).
- The trade-offs between MobileNetV2 (designed for mobile/edge efficiency) versus ResNet or Vision Transformers (ViT) in terms of accuracy and latency.

## 7. Tomorrow's Plan (Day 5: Sentiment Analysis Pipeline)
- Install `pandas`, `scikit-learn`, and `nltk`.
- Prepare a labeled dataset of text reviews (positive, negative, neutral) for TF-IDF feature extraction and sentiment classification.