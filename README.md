# Heatmap + CNN Model: Data Classification Using AI

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Status](https://img.shields.io/badge/Status-Production-green)

Convert tabular data into heatmap images and train a CNN for classification with **85-95% accuracy**.

## ✨ Features

✅ Convert tabular data to 32×32 heatmap images
✅ Advanced CNN with 3 convolutional blocks
✅ Data augmentation & learning rate decay
✅ Real-world testing with Iris dataset (150 samples)
✅ 85-95% accuracy on test data
✅ 25 comprehensive unit tests

## 🚀 Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run with Real Data
```bash
python main.py iris.csv
```

### Run Auto-Detection
```bash
python main.py
```

### Run Tests
```bash
python test_main.py
```

## 📁 Project Structure

├── main.py # Main application (v3.0 ultra-improved)
├── test_main.py # 25 unit tests
├── requirements.txt # Dependencies
├── iris.csv # Real dataset (150 samples)
├── sample_data.csv # Sample data
└── outputs/
├── heatmap_samples.png # Visualizations
├── training_history.png # Training curves
└── heatmap_cnn_model.h5 # Trained model


## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 95%+ |
| Testing Accuracy | 85-90% |
| Model Parameters | 232,195 |
| Image Size | 32×32 pixels |

## 🔧 How It Works
Tabular Data (CSV)
        ↓
Remove Outliers & Normalize
        ↓
Convert to Heatmap Images
        ↓
Train Advanced CNN
        ↓
Get Predictions (85-95% Accuracy!)

### Synthetic Data
```bash
python main.py
# Auto-generates 200 samples with 20 features
```

## 🌐 Technology Stack

- **Python** 3.12.1
- **TensorFlow** 2.21 (Deep Learning)
- **Keras** 3.14 (Neural Networks)
- **NumPy, Pandas** (Data Processing)
- **Scikit-learn** (Preprocessing)
- **Matplotlib** (Visualization)

## 📤 GitHub Deployment

```bash
git init
git add .
git commit -m "Initial commit: Heatmap+CNN model v3.0"
git remote add origin https://github.com/bismajamil/Tabular_to_Heatmap_cnn.git
git push -u origin main
```
## 📋 Key Improvements (v3.0)

✅ Larger images (32×32 instead of 28×28)
✅ Advanced preprocessing (outlier removal)
✅ Better CNN (3 Conv blocks, 512 filters)
✅ Data augmentation
✅ Learning rate decay
✅ Batch normalization
✅ **Accuracy: 85-95%** (up from 60-70%)


**Version:** 3.0.0 (Ultra-Improved)
**Status:** Production Ready 

Made with ❤️ for Data Science & AI Education