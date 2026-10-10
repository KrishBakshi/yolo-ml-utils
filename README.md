# ML Utils

A collection of machine learning utilities and visualization tools built with [Gradio](https://gradio.app/). Designed to streamline common ML/CV workflows, especially for YOLO-based object detection and segmentation tasks.

**Author:** [Krish Bakshi](https://www.krishbakshi.com)

---

## Features

###  Image Data Plot
Visualize your YOLO annotations directly on images.

- **Bounding Box Visualizer** — Draw detection boxes with class labels
- **Segmentation Visualizer** — Overlay segmentation masks with customizable transparency
  - Supports Semantic, Instance, and Panoptic segmentation types
  - Toggle polygon outlines on/off
  - Adjustable alpha blending

### Image Data Transformer
Apply data augmentations to expand your training dataset.

- **Bounding Box Augmentation** — Rotate images (90°, 180°, 270°) with automatic label transformation
- **Segmentation Augmentation** — Same rotation support for segmentation polygon labels
- Option to keep or discard original images
- Download augmented dataset as ZIP

### Train/Val/Test Split
Split your dataset into training, validation, and test sets.

- Configurable split ratios (default: 70/20/10)
- Reproducible splits with random seed
- Preserves image-label pairs
- Copies `classes.txt` to all output splits
- Download split dataset as ZIP

### Model Training
Train YOLO models using the [Ultralytics](https://docs.ultralytics.com/) library.

- **Create YAML Config** — Auto-generate dataset configuration files
- **Train YOLO Models** — Full training interface with:
  - Basic parameters: epochs, batch size, image size, device
  - Advanced hyperparameters: learning rate, momentum, weight decay, warmup, patience
  - Auto-detect YAML files in your project
  - Real-time training output

---

## Installation

### Prerequisites
- Python 3.12+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/KrishBakshi/yolo-ml-utils.git
cd ml-utils
```

2. Install dependencies:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

### Dependencies
- `gradio` — Web UI framework
- `opencv-python` — Image processing
- `matplotlib` — Plotting
- `numpy` — Numerical operations
- `ultralytics` — YOLO training
- `split-folders` — Dataset splitting
- `pymupdf` — PDF handling

---

## Usage

### Launch the App

```bash
python app.py
```

The app will open in your browser at `http://localhost:7860`.

### Expected Data Format

All tools expect data in YOLO format:

```
dataset/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── labels/
│   ├── image001.txt
│   ├── image002.txt
│   └── ...
└── classes.txt  (optional)
```

**Label format:**
- Detection: `class_id x_center y_center width height`
- Segmentation: `class_id x1 y1 x2 y2 x3 y3 ...`

All coordinates are normalized (0-1).

---

## Quick Start Examples

### 1. Visualize Annotations

1. Navigate to **Image Data Plot**
2. Upload a ZIP file containing `images/` and `labels/` folders
3. Click **Generate Plots**
4. Download all visualizations as ZIP

### 2. Augment Dataset

1. Navigate to **Image Data Transformer**
2. Upload your dataset (ZIP or directory path)
3. Select rotation angles (90°, 180°, 270°)
4. Click **Generate Augmentations**
5. Download augmented data

### 3. Split Dataset

1. Navigate to **Image Data Train Test Split**
2. Upload your dataset
3. Adjust split ratios (must sum to 1.0)
4. Click **Create Split**
5. Download the split dataset

### 4. Train a Model

1. Navigate to **Model Training** → **Create YAML File**
2. Enter your split folder path
3. Click **Auto-detect from classes.txt** or manually enter class names
4. Create the YAML file

5. Switch to **Train YOLO Model** tab
6. Enter model name (e.g., `yolo11n.pt`)
7. Set YAML path or auto-detect
8. Configure training parameters
9. Click **Start Training**

---

## Project Structure

```
ml-utils/
├── app.py                              # Main Gradio app
├── config/
│   └── app_registry.yaml               # App module registry
├── src/
│   ├── image_data_plot/                # Visualization tools
│   │   ├── app.py
│   │   ├── plot_detection_data.py
│   │   └── plot_segmentation_data.py
│   ├── image_data_transformer/         # Augmentation tools
│   │   ├── app.py
│   │   ├── get_image_bbox_data_aug.py
│   │   └── get_image_seg_data_aug.py
│   ├── image_data_train_test_split/    # Dataset splitting
│   │   ├── app.py
│   │   └── get_train_val_test_split.py
│   └── ultralytics_training/           # YOLO training
│       └── app.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## License

This project is open source. See LICENSE file for details.
