# Space Debris Detection Pipeline

![Space Debris Detection](ASSETS/readme_header.jpg)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![Blender](https://img.shields.io/badge/Blender-4.2-purple.svg)](https://www.blender.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Highlights

- **End-to-end pipeline** for space debris detection using synthetic images, computer vision, and deep learning, designed to address the lack of labeled datasets and high cost of real-world data collection.
- **Scalable synthetic image generation** with Blender’s Python API. 1000 images? Yes! 10,000 images? No problem! 100,000 images? Absolutely. Easily generates labelled, photorealistic training data to simulate space debris in orbit.
- **Advanced domain randomisation and image augmentation techniques** to replicate real-world space imaging challenges.
- **Training YOLO models** (YOLOv8n, YOLOv11n, YOLOv12n) with full Weights & Biases integration for experiment tracking and benchmarking. 
- **Optimized for lightweight deployment,** with a focus on models suited for constrained environments like Active Debris Removal spacecraft (limited volume, mass, power, and cost).
- **Real-world inference capability,** validating that models trained on synthetic images can successfully detect debris in real-world space imagery.

## 📋 Overview

The Space Debris Detection Pipeline is a complete, modular framework for detecting and classifying space debris using synthetic imagery, computer vision, and deep learning. Designed to address the scarcity of labeled space debris datasets and the high cost of real data collection, it integrates scalable synthetic image generation with Blender, advanced augmentation techniques, and a streamlined detection system built around the YOLO family of models. The pipeline enables training on large-scale, photorealistic datasets and demonstrates real-world inference capability, paving the way for robust, resource-efficient debris detection in Active Debris Removal missions.

## ✍️ Author
I'm Jai, and I built this project out of a deep fascination with space technology and a curiosity about how AI can be applied to solve real-world problems. It started as a university project but quickly became a personal mission to explore how synthetic data and deep learning can work together in a practical, meaningful way. 

Along the way, I learned a ton - about space, technology, and myself. I saw firsthand what it takes to turn a single idea into a fully working system (woah). I hope this repository helps others on a similar path or sparks new ideas in the process.

### 🚀 Motivation

Space debris poses an escalating threat to satellites, human missions, and the long-term sustainability of space operations. As thousands of objects continue to accumulate in Earth’s orbit, the risk of collisions grows, and so does the urgency to monitor and manage debris effectively. This project was inspired by my fascination with space technology and computer vision, and by a desire to contribute meaningful tools to the field of space situational awareness using modern AI.

Nearly 50 years ago, NASA scientist Donald Kessler warned: “Keep cluttering up space, and we'll eventually lock ourselves out of it.” That warning feels more real than ever today. With the high costs and scarcity of real-world debris data, I saw an opportunity to create a scalable, affordable solution: one that blends synthetic image generation and deep learning to help push the boundaries of real-time space debris detection.

### 🔍 Project Structure

```
space-debris-detection-pipeline/
├── ASSETS/                  # General assets (e.g., fonts, background images) for documentation and plots
├── BLENDER_ASSETS/          # Blender resources for synthetic image generation
│   ├── blender_files/       # Blender model (.blend) and Python API script for automated rendering
│   └── debris_models.zip/   # 3D debris models (.obj, .fbx, .glb) used in scene setup
├── ML_ASSETS/               # Machine learning assets for dataset storage and model outputs
│   └── dataset/             # Dataset directories and results
│       ├── Debris/          # Raw dataset (images and YOLO-format annotations)
│       ├── Debris_Scaled/   # Scaled dataset, augmentations, YOLO configs, and training outputs
│       ├── Results/         # Evaluation results, plots, and confusion matrices
│       └── ...              # Additional metadata and coordinate files
├── TECHNICAL_DOCUMENT/      # Technical report
├── environment.yml          # Conda environment specification for full reproducibility
├── main.ipynb               # Jupyter notebook implementing the full detection pipeline
├── main.py                  # Python script version of the pipeline (standalone execution)
└── requirements.txt         # Python package requirements (for pip-based setups)
```

## 🖼️ Examples

### Synthetic Test Image Examples
<p float="left">
  <img src="ML_ASSETS/dataset/Debris_Scaled/test/images/000193.jpg" width="200" alt="Test Image 1"/>
  <img src="ML_ASSETS/dataset/Debris_Scaled/test/images/000590.jpg" width="200" alt="Test Image 2" />
  <img src="ML_ASSETS/dataset/Debris_Scaled/test/images/000598.jpg" width="200" alt="Test Image 3" />
</p>

### Real-world Test Image Examples
<p float="left">
  <img src="ML_ASSETS/dataset/Debris_Scaled/test/images/Hubble_Real_Life_1.jpg" width="200" alt="Real Image 1"/>
  <img src="ML_ASSETS/dataset/Debris_Scaled/test/images/Hubble_Real_Life_2.jpg" width="200" alt="Real Image 2" />
  <img src="ML_ASSETS/dataset/Debris_Scaled/test/images/Hubble_Real_Life_3.jpg" width="200" alt="Real Image 3" />
</p>

### Inference Examples
Synthetic images with bounding box predictions (YOLOv8n, YOLOv11n, and YOLOv12n):
<p float="left">
  <img src="ML_ASSETS\dataset\Debris_Scaled\test\images\inference\Inference Run 9 yolov8n_best_weights (Mar-30-2025 , 21-30-22)\000193.jpg" width="200" alt="Inference Image 1"/>
  <img src="ML_ASSETS\dataset\Debris_Scaled\test\images\inference\Inference Run 9 yolov8n_best_weights (Mar-30-2025 , 21-30-22)\000590.jpg" width="200" alt="Inference Image 2" />
  <img src="ML_ASSETS\dataset\Debris_Scaled\test\images\inference\Inference Run 9 yolov8n_best_weights (Mar-30-2025 , 21-30-22)\000598.jpg" width="200" alt="Inference Image 3" />
</p>
Real-world images with bounding box predictions (YOLOv8n, YOLOv11n, and YOLOv12n):
<p float="left">
  <img src="ML_ASSETS\dataset\Debris_Scaled\test\images\inference\Inference Run 9 yolov8n_best_weights (Mar-30-2025 , 21-30-22)\Hubble_Real_Life_1.jpg" width="200" alt="Inference Image 1"/>
  <img src="ML_ASSETS\dataset\Debris_Scaled\test\images\inference\Inference Run 9 yolov8n_best_weights (Mar-30-2025 , 21-30-22)\Hubble_Real_Life_2.jpg" width="200" alt="Inference Image 2" />
  <img src="ML_ASSETS\dataset\Debris_Scaled\test\images\inference\Inference Run 9 yolov8n_best_weights (Mar-30-2025 , 21-30-22)\Hubble_Real_Life_3.jpg" width="200" alt="Inference Image 3" />
</p>

## 🛠️ Installation

### Prerequisites

- Python 3.12 or higher
- CUDA-compatible GPU (for training)
- Blender 4.2 (for data generation)

### Setup with Conda

```bash
# Clone the repository
git clone https://github.com/jaikr-dev/space-debris-detection-pipeline.git
cd space-debris-detection-pipeline

# Create and activate conda environment
conda env create -f environment.yml
conda activate space_debris_detection_pipeline
```

### Setup with pip

```bash
# Clone the repository
git clone https://github.com/jaikr-dev/space-debris-detection-pipeline.git
cd space-debris-detection-pipeline

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Data Generation with Blender

The pipeline includes a Blender-based synthetic data generator that creates images of space debris with annotations for training:

```python
# Start the Blender rendering process
run_blender_with_monitoring()
```

### Data Preprocessing

Preprocess and scale the generated images for model training:

```python
# Scale images to the required dimensions
scale_images(new_size=(640, 640))

# Apply augmentations to simulate space imaging conditions
apply_augmentations()
```

### Model Training

Train a YOLOv8 model on the prepared dataset:

```python
# Train the model with Weights & Biases integration
train_and_log_model(epochs=100, batch_size=16)
```

### Inference

Run inference on new images using a trained model:

```python
# Perform inference using the best weights
results = run_inference(
    model_weight_path="ML_ASSETS/dataset/Debris_Scaled/Run 9 yolov8n (Mar-25-2025 , 09-49-56)/weights/best.pt",
    confidence_threshold=0.80
)
```

## 🧪 Running Tests

The project includes validation and testing capabilities:

```python
# Validate the model on the test set
model = YOLO("ML_ASSETS/dataset/Debris_Scaled/Run 9 yolov8n (Mar-25-2025 , 09-49-56)/weights/best.pt")
results = model.val()

# Extract and analyze confusion matrix
extract_confusion_matrix(model, run_dir)
```

## 📊 Results

The pipeline achieves the following performance metrics:

- **YOLOv8n**: mAP50-95 = 0.63, Precision = 0.80, Recall = 0.69
- **YOLOv11n**: mAP50-95 = 0.75, Precision = 0.82, Recall = 0.85
- **YOLOv12n**: mAP50-95 = 0.69, Precision = 0.71, Recall = 0.77

## 🐛 Known Issues & Limitations

- The pipeline currently supports detection of 4 specific debris types: Satellite, Envisat, Hubble, and Falcon 9 F&S
- Real-world performance may vary depending on image quality and lighting conditions
- The Blender data generation process can be time-consuming for large datasets

## 🔮 Future Work

- Expand the debris types to include smaller orbital fragments
- Implement tracking capabilities for moving debris
- Add support for multi-frame analysis from video sources
- Develop a web interface for real-time debris detection
- Integrate with space situational awareness databases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [Blender](https://www.blender.org/) for 3D modeling and rendering capabilities
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Albumentations](https://albumentations.ai/) for image augmentation techniques