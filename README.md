# 🔆 Solar Module Health Monitoring via EL Image Segmentation

A deep learning-based pipeline for automated **fault detection and classification** in solar PV modules using **electroluminescence (EL) images**. The system performs semantic segmentation to identify 29+ defect and feature classes across multiple datasets with varying formats and gray-coded masks.

---

## 📁 Dataset Structure

```
dataset_YYYYMMDD/
├── el_images_train/
├── el_images_val/
├── el_images_test/
├── el_masks_train/
├── el_masks_val/
├── el_masks_test/
└── ListOfClassesAndColorCodes.csv
```

- Grayscale masks have class-specific pixel values defined in the CSV file.
- Datasets include pre-augmented images (flip, rotate, mirror).

---

## ⚙️ YAML-Driven Configuration

 `config.yaml`:

```yaml
data:
  root_dir: ./datasets
  version: dataset_20221008
  img_size: 512
  batch_size: 4
  use_rgb_masks: false

train:
  epochs: 50
  lr: 0.0001
  use_class_weights: true
  class_weights: [1.0, 3.5, 5.0, ...]

experiment:
  name: model_run
  output_dir: results/
```

---

## 🧠 Model

Supports:

- ✅ DeepLabV3+ (default)
- 🧪 UNet++

Outputs logits over all defect classes per pixel.

---

## 🚀 Running the Pipeline

### 🔹 Train

```bash
python main.py --config config.yaml
```

### 🔹 Test

```bash
python test.py --config config.yaml
```

Results are saved in:

```
results/
├── images/   # Input EL images
├── masks/    # Predicted RGB masks
```

### 🔹 Inference on Single Image

```bash
python run.py --image_path path/to/image.png --config config.yaml
```

---

## 📊 Performance Metrics

| Metric        | Value (example)             |
| ------------- | --------------------------- |
| Train Loss    | ↓ Decreases steadily        |
| mIoU          | ~42.7% (varies by dataset)  |
| Per-class IoU | Visualized per epoch        |
| Best Model    | Saved automatically         |

---

## 🖼️ Sample Output

| EL Image                       | Predicted Mask               |
| ----------------------------- | ---------------------------- |
| `results/images/image_001.png` | `results/masks/mask_001.png` |

---

## 🔧 Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch, torchvision
- numpy, pandas, Pillow, tqdm, PyYAML

---

## 📚 References

- Dataset: [TheMakiran/BenchmarkELimages](https://github.com/TheMakiran/BenchmarkELimages)
- Paper: [ScienceDirect, 2023](https://www.sciencedirect.com/science/article/pii/S2772941923000017)

---

