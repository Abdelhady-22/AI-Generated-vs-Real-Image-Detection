# 🔍 AI-Generated Image Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/)

> Distinguishing real photographs from AI-generated images using state-of-the-art deep learning architectures. This project implements both custom Vision Transformers and transfer learning with ResNet50V2, achieving up to **95.43% accuracy** on a balanced dataset of 200,000 images.

---

## 📋 Table of Contents

- [Features](#-features)
- [Model Performance](#-model-performance)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Methodology](#-methodology)
- [Notebooks](#-notebooks)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 🎯 Dual Architecture Approach
- **Vision Transformer (ViT)**: Custom implementation from scratch with patch embeddings and multi-head attention
- **ResNet50V2**: Transfer learning from ImageNet pre-trained weights with custom classification head

### 📊 Comprehensive Analysis
- Detailed confusion matrices and classification reports
- Training history visualization (loss, accuracy, precision, recall, AUC)
- Per-class performance metrics
- Model comparison and benchmarking

### 🚀 Production-Ready Pipeline
- Efficient TensorFlow data loading with prefetching
- Automated data splitting (train/val/test)
- Model checkpointing and early stopping
- Learning rate scheduling

### 🔬 Robust Evaluation
- Multiple metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Balanced test set evaluation (15,000 images per class)
- Specificity and sensitivity analysis

---

## 🎯 Model Performance

### Quick Comparison

| Model | Test Accuracy | Test Precision | Test Recall | Test AUC | Parameters | Training Time/Epoch |
|-------|---------------|----------------|-------------|----------|------------|---------------------|
| **ResNet50V2** | **95.43%** | **95.34%** | **95.53%** | **99.03%** | 558k (trainable) | ~12 min |
| **Vision Transformer** | 91.14% | 92.04% | 90.07% | 97.20% | 28.9M | ~43 min |

### Best Model: ResNet50V2 Transfer Learning 🏆

#### Test Set Performance (30,000 images)
- **Overall Accuracy**: 95.43%
- **Precision**: 95.34% (low false positive rate)
- **Recall**: 95.53% (high detection rate)
- **F1-Score**: 95.43%
- **AUC-ROC**: 99.03% (excellent class separation)
- **Specificity**: 95.33%
- **Test Loss**: 0.1222

#### Confusion Matrix Breakdown
| Predicted → | Fake | Real |
|-------------|------|------|
| **Actual Fake** | 14,299 (TN) | 701 (FP) |
| **Actual Real** | 671 (FN) | 14,329 (TP) |

**Total Errors**: 1,372 / 30,000 = **4.57% error rate**

### Vision Transformer Performance

#### Test Set Results
- **Overall Accuracy**: 91.14%
- **Precision**: 92.04%
- **Recall**: 90.07%
- **F1-Score**: 91.04%
- **AUC-ROC**: 97.20%
- **Test Loss**: 0.2198

---

## 🎬 Demo

### Example Predictions

```python
# Load trained model
model = load_model('ResNet_best_model.keras')

# Predict on new image
image = load_and_preprocess_image('suspicious_image.jpg')
prediction = model.predict(image)

if prediction > 0.5:
    print(f"REAL IMAGE (Confidence: {prediction[0][0]*100:.2f}%)")
else:
    print(f"FAKE IMAGE (Confidence: {(1-prediction[0][0])*100:.2f}%)")
```

### Typical Results
- ✅ **Real photo detection**: 95.53% success rate
- ✅ **Fake image detection**: 95.33% success rate
- ✅ **Balanced performance**: No bias toward either class

---

## 📊 Dataset

### Overview

This project uses a carefully curated dataset combining real and AI-generated images:

| Dataset | Source | Type | Count |
|---------|--------|------|-------|
| **COCO 2017** | Microsoft | Real photographs | 100,000 |
| **DiffusionDB** | Part 0001-0100 | AI-generated (diffusion models) | 100,000 |
| **Total** | - | Balanced binary classification | **200,000** |

### Dataset Split

| Split | Fake Images | Real Images | Total | Percentage |
|-------|-------------|-------------|-------|------------|
| **Training** | 70,000 | 70,000 | 140,000 | 70% |
| **Validation** | 15,000 | 15,000 | 30,000 | 15% |
| **Testing** | 15,000 | 15,000 | 30,000 | 15% |

### Data Sources

1. **COCO 2017 (Real Images)**
   - [Kaggle Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
   - Real-world photographs spanning 80+ object categories
   - Natural scenes, people, animals, objects
   - High-quality, professionally captured images

2. **DiffusionDB (Fake Images)**
   - [Kaggle Dataset](https://www.kaggle.com/datasets/ammarali32/diffusiondb-2m-part-0001-to-0100-of-2000)
   - AI-generated images from Stable Diffusion and other diffusion models
   - Diverse prompts and styles
   - Represents state-of-the-art generative AI output

### Preprocessing

All images undergo standardized preprocessing:
- **Resize**: 224 × 224 pixels (standard for ResNet and ViT)
- **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
- **Format**: RGB (3 channels)
- **Batch Size**: 32 images per batch
- **Data Type**: Float32

---

## 📁 Project Structure

```
AI-GENERATED-IMAGE-DETECTION/
│
├── notebooks/
│   ├── vision_transformer.ipynb        # Custom ViT implementation
│   ├── resnet50v2_transfer.ipynb       # ResNet transfer learning
│   
│
├── models/
│   ├── vit_best_model.keras            # Best ViT checkpoint
│   ├── ResNet_best_model.keras         # Best ResNet checkpoint
│   
│
├── results/
│   ├── vit_results/
│   │   ├── confusion_matrix.png
│   │   ├── training_history.png
│   │   └── classification_report.txt
│   ├── resnet_results/
│   │   ├── confusion_matrix.png
│   │   ├── training_history.png
│   │   └── classification_report.txt
│   └── Data_Visualization/
│       ├── fake_images_1.png
│       ├── fake_images_2.png
│       ├── real_images_1.png
│       ├── real_images_2.png
│       ├── count.png
│
├── data/
│   ├── train/
│   │   ├── fake/                       # Symbolic links to training fake images
│   │   └── real/                       # Symbolic links to training real images
│   ├── val/
│   │   ├── fake/
│   │   └── real/
│   └── test/
│       ├── fake/
│       └── real/
│
├── src/
│   ├── data_preprocessing.py           # Data loading and preprocessing
│   ├── model_training.py               # Training utilities
│   ├── evaluation.py                   # Evaluation metrics
│   └── visualization.py                # Plotting functions
│
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment
├── README.md                           # This file
├── LICENSE                             # MIT License
└── .gitignore                          # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM
- 50GB+ free disk space

### Option 1: Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/Abdelhady-22/AI-Generated-vs-Real-Image-Detection.git
cd AI-Generated-vs-Real-Image-Detection

# Create conda environment
conda env create -f environment.yml
conda activate fake-image-detection

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} | GPU:', tf.config.list_physical_devices('GPU'))"
```

### Option 2: Using pip

```bash
# Clone repository
git clone https://github.com/Abdelhady-22/AI-Generated-vs-Real-Image-Detection.git
cd AI-Generated-vs-Real-Image-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### Download Datasets

#### Method 1: Using Kaggle API

```bash
# Install kagglehub
pip install kagglehub

# Download datasets programmatically
python << EOF
import kagglehub

# Download COCO 2017
coco_path = kagglehub.dataset_download('awsaf49/coco-2017-dataset')
print(f'COCO downloaded to: {coco_path}')

# Download DiffusionDB
diffusion_path = kagglehub.dataset_download('ammarali32/diffusiondb-2m-part-0001-to-0100-of-2000')
print(f'DiffusionDB downloaded to: {diffusion_path}')
EOF
```

#### Method 2: Manual Download

1. Download COCO 2017: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
2. Download DiffusionDB: https://www.kaggle.com/datasets/ammarali32/diffusiondb-2m-part-0001-to-0100-of-2000
3. Extract to `data/raw/` directory

---

## 💻 Usage

### Quick Start

```python
# Train ResNet50V2 (recommended)
python notebooks/resnet50v2_transfer.ipynb

# Or train Vision Transformer
python notebooks/vision_transformer.ipynb
```

### Step-by-Step Training

#### 1. Data Preparation

```python
import os
from pathlib import Path

# Define paths
CLASS0_DIR = "data/raw/diffusiondb/"  # Fake images
CLASS1_DIR = "data/raw/coco2017/train2017/"  # Real images
OUTPUT_DIR = "data/processed/"

# Create train/val/test splits
from src.data_preprocessing import create_splits

create_splits(
    fake_dir=CLASS0_DIR,
    real_dir=CLASS1_DIR,
    output_dir=OUTPUT_DIR,
    train_size=70000,
    val_size=15000,
    test_size=15000,
    seed=42
)
```

#### 2. Train ResNet50V2

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, Model

# Load data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/processed/train',
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)

# Build model
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC()]
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint('models/best_model.keras')
    ]
)
```

#### 3. Evaluate Model

```python
# Load test data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/processed/test',
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)

# Evaluate
results = model.evaluate(test_ds)
print(f"Test Accuracy: {results[1]*100:.2f}%")

# Generate predictions
from src.evaluation import evaluate_model

evaluate_model(model, test_ds, save_path='results/')
```

### Inference on New Images

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load trained model
model = load_model('models/ResNet_best_model.keras')

def predict_image(image_path, threshold=0.5):
    """
    Predict if an image is real or AI-generated.
    
    Args:
        image_path: Path to image file
        threshold: Classification threshold (default 0.5)
    
    Returns:
        dict: Prediction results
    """
    # Load and preprocess
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret
    is_real = prediction > threshold
    confidence = prediction if is_real else (1 - prediction)
    
    return {
        'prediction': 'REAL' if is_real else 'FAKE',
        'confidence': confidence * 100,
        'raw_score': prediction
    }

# Example usage
result = predict_image('test_image.jpg')
print(f"{result['prediction']} (Confidence: {result['confidence']:.2f}%)")
```

---

## 🏗️ Model Architectures

### ResNet50V2 Transfer Learning

```
Input (224, 224, 3)
        ↓
┌──────────────────────────────┐
│  ResNet50V2 Base (Frozen)    │
│  - Pre-trained on ImageNet   │
│  - 23.5M parameters          │
│  - Feature extraction        │
└──────────────────────────────┘
        ↓
GlobalAveragePooling2D → (2048,)
        ↓
Dense(256, relu) → BatchNorm → Dropout(0.5)
        ↓
Dense(128, relu) → BatchNorm → Dropout(0.3)
        ↓
Dense(1, sigmoid) → [0, 1]
        ↓
Output: 0 = Fake, 1 = Real
```

**Key Features:**
- **Transfer Learning**: Leverages ImageNet knowledge
- **Frozen Base**: Only trains classification head (558k params)
- **Regularization**: BatchNorm + Dropout prevents overfitting
- **Efficiency**: 3× faster training than ViT

### Vision Transformer (ViT)

```
Input Image (224, 224, 3)
        ↓
┌──────────────────────────────┐
│  Patch Embedding (16×16)     │
│  → 196 patches × 384 dim     │
└──────────────────────────────┘
        ↓
CLS Token + Positional Encoding
        ↓
┌──────────────────────────────┐
│  Transformer Encoder × 6     │
│  ┌──────────────────────┐   │
│  │ Layer Normalization  │   │
│  │ Multi-Head Attention │   │
│  │ (6 heads)            │   │
│  │ Residual Connection  │   │
│  ├──────────────────────┤   │
│  │ Layer Normalization  │   │
│  │ MLP (384→1536→384)   │   │
│  │ Residual Connection  │   │
│  └──────────────────────┘   │
└──────────────────────────────┘
        ↓
Extract CLS Token → (384,)
        ↓
Dense(512, gelu) → Dropout(0.3)
        ↓
Dense(1, sigmoid) → [0, 1]
```

**Key Features:**
- **Attention Mechanism**: Learns spatial relationships
- **Patch-Based**: Processes 16×16 image patches
- **Deep Architecture**: 6 transformer blocks
- **Large Capacity**: 28.9M parameters

---

## 📈 Results

### Training Curves

#### ResNet50V2
- **Convergence**: Best model at epoch 12
- **Training Time**: ~3.2 hours (15 epochs)
- **Validation Accuracy**: 95.32% (peak)
- **Minimal Overfitting**: Train-val gap < 0.5%

#### Vision Transformer
- **Convergence**: Best model at epoch 10
- **Training Time**: ~9 hours (13 epochs)
- **Validation Accuracy**: 91.41% (peak)
- **Slight Overfitting**: Train-val gap ~1.5%

### Performance Metrics Comparison

| Metric | ResNet50V2 | ViT | Winner |
|--------|------------|-----|--------|
| **Accuracy** | 95.43% | 91.14% | ResNet 🏆 |
| **Precision** | 95.34% | 92.04% | ResNet 🏆 |
| **Recall** | 95.53% | 90.07% | ResNet 🏆 |
| **F1-Score** | 95.43% | 91.04% | ResNet 🏆 |
| **AUC-ROC** | 99.03% | 97.20% | ResNet 🏆 |
| **Specificity** | 95.33% | 92.21% | ResNet 🏆 |
| **Training Speed** | 12 min/epoch | 43 min/epoch | ResNet 🏆 |
| **Parameters** | 24.1M (2.3% trainable) | 28.9M (100% trainable) | ResNet 🏆 |

### Key Insights

1. **Transfer Learning Dominates**: ResNet50V2 outperforms custom ViT by 4.3% accuracy
2. **Efficiency Matters**: ResNet trains 3× faster with fewer parameters
3. **Balanced Performance**: Both models show minimal class bias
4. **Excellent Generalization**: High validation → test consistency
5. **Production Ready**: ResNet achieves 95%+ accuracy with fast inference

---

## 🔬 Methodology

### Data Collection
1. **Real Images**: 100k from COCO 2017 (natural photographs)
2. **Fake Images**: 100k from DiffusionDB (AI-generated)
3. **Balanced Split**: 70k train / 15k val / 15k test per class

### Preprocessing Pipeline
1. **Resize**: All images → 224×224 pixels
2. **Normalization**: Pixel values [0, 255] → [0, 1]
3. **Batching**: Groups of 32 images
4. **Prefetching**: Overlap data loading with training

### Training Strategy
1. **Transfer Learning** (ResNet):
   - Freeze pre-trained ImageNet weights
   - Train only custom classification head
   - Fine-tune learning rate: 1e-4
   
2. **From Scratch** (ViT):
   - Random weight initialization
   - Full model training
   - Learning rate: 1e-4

### Optimization
- **Optimizer**: Adam with default parameters
- **Loss Function**: Binary cross-entropy
- **Callbacks**:
  - Early stopping (patience=3)
  - Model checkpointing (save best)
  - Learning rate reduction (factor=0.5, patience=2)

### Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Test Set**: 30,000 unseen images
- **Confusion Matrix**: Detailed error analysis
- **Cross-Validation**: Consistent train-val-test splits

---

## 📓 Notebooks

### 1. Vision Transformer Implementation
**File**: `notebooks/vision_transformer.ipynb`

**Contents**:
- Custom ViT architecture from scratch
- Patch embedding and positional encoding
- Multi-head self-attention mechanisms
- Training on 200k images
- Comprehensive evaluation

**Key Results**: 91.14% test accuracy, 97.20% AUC

### 2. ResNet50V2 Transfer Learning
**File**: `notebooks/resnet50v2_transfer.ipynb`

**Contents**:
- Transfer learning from ImageNet
- Custom classification head design
- Efficient training (3× faster than ViT)
- Superior performance metrics

**Key Results**: 95.43% test accuracy, 99.03% AUC

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
```bash
git clone https://github.com/Abdelhady-22/AI-Generated-vs-Real-Image-Detection.git
cd AI-Generated-vs-Real-Image-Detection
```

2. **Create a feature branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Make your changes**
   - Add new models or improve existing ones
   - Enhance documentation
   - Fix bugs or optimize code

4. **Run tests**
```bash
python -m pytest tests/
```

5. **Commit your changes**
```bash
git commit -m "Add amazing feature"
```

6. **Push to your fork**
```bash
git push origin feature/amazing-feature
```

7. **Open a Pull Request**

### Contribution Ideas

- 🎯 Implement additional architectures (EfficientNet, ConvNeXt, Swin Transformer)
- 📊 Add ensemble methods for improved accuracy
- 🔍 Implement Grad-CAM for model interpretability
- 📱 Create web interface for easy inference
- 📈 Add support for video deepfake detection
- 🌍 Extend to multi-class fake image detection
- 🧪 Add unit tests and integration tests
- 📝 Improve documentation and tutorials

---

## 📜 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{ai_generated_image_detection_2024,
  author = {Abdelhady Ali Mohamed},
  title = {AI-Generated Image Detection using Deep Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Abdelhady-22/AI-Generated-vs-Real-Image-Detection},
  note = {ResNet50V2 Transfer Learning achieving 95.43\% accuracy}
}
```

### Related Papers

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" - Vision Transformer foundation
2. He et al. (2016). "Deep Residual Learning for Image Recognition" - ResNet architecture
3. Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models" - Stable Diffusion background

---

## 📄 License

This project is licensed under the Apache License - see the [Apache](LICENSE) file for details.

```
Apache License

Copyright (c) 2025 [Abdelhady Ali]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full MIT License text...]
```

---

## 🙏 Acknowledgments

### Datasets
- **COCO 2017**: Microsoft COCO Team for high-quality real image dataset
- **DiffusionDB**: Researchers at UC Berkeley for AI-generated image dataset

### Frameworks & Tools
- **TensorFlow/Keras**: Deep learning framework
- **Kaggle**: Computational resources and platform
- **scikit-learn**: Machine learning utilities
- **OpenCV**: Image processing library

### Inspiration
- Vision Transformer paper by Google Research
- ResNet architecture by Microsoft Research
- AI safety research community

### Community
- Stack Overflow and GitHub communities
- Kaggle discussion forums
- TensorFlow documentation contributors

---

## 📞 Contact

**Author**: [Abdelhady Ali]

- 📧 Email: abdulhadi2322005@gmail.com
- 💼 LinkedIn: [My LinkedIn](www.linkedin.com/in/abdelhady-ali-940761316)
- 🐙 GitHub: [MY username](https://github.com/Abdelhady-22)

### Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Abdelhady-22/AI-Generated-vs-Real-Image-Detection/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/Abdelhady-22/AI-Generated-vs-Real-Image-Detection/discussions)
- 📧 **General Inquiries**: abdulhadi2322005@gmail.com

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. The models are designed to detect AI-generated images but should not be used as the sole basis for:

- Legal proceedings or evidence authentication
- Journalistic verification without additional fact-checking
- Medical or scientific image validation
- Any decision with significant consequences

**Important Notes**:
1. Model performance may vary on images from newer generative models
2. Adversarial attacks can fool detection systems
3. Always combine automated detection with human expertise
4. Regular model updates are needed as generative AI evolves

---

## 🚀 Future Roadmap

### Short-term (Q1-Q2 2024)
- [ ] Add EfficientNetV2 and ConvNeXt models
- [ ] Implement Grad-CAM visualization
- [ ] Create Flask/FastAPI web interface
- [ ] Add Docker containerization
- [ ] Improve documentation with video tutorials

### Medium-term (Q3-Q4 2024)
- [ ] Ensemble multiple models for 96%+ accuracy
- [ ] Support for video deepfake detection
- [ ] Multi-class detection (identify generation method)
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Continuous learning pipeline

### Long-term (2025+)
- [ ] Real-time browser extension
- [ ] Integration with social media platforms
- [ ] Advanced adversarial robustness
- [ ] Multi-modal detection (image + metadata)
- [ ] Research paper publication

---

## 📊 Statistics

- **Total Images Processed**: 200,000
- **Training Examples**: 140,000
- **Test Examples**: 30,000
- **Model Parameters**: 24-29 million
- **Training Time**: 3-9 hours (GPU)
- **Inference Speed**: ~50 images/second (GPU)
- **Project Stars**: ⭐ (Star this repo!)

---

<div align="center">

**🌟 Star this repository if you find it helpful! 🌟**

Made with ❤️ for AI safety and transparency

[⬆ Back to Top](#-ai-generated-image-detection-using-deep-learning)

</div>
