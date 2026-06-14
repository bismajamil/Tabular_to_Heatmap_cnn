# Heatmap + CNN Model
## Convert Tabular Data to Computer Vision

A complete machine learning pipeline that converts tabular (spreadsheet) data into heatmap images and trains a Convolutional Neural Network (CNN) for classification.

![Status](https://img.shields.io/badge/Status-Production-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Testing in VS Code](#testing-in-vs-code)
- [Results & Output](#results--output)
- [GitHub Deployment](#github-deployment)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

✅ **Convert Any Tabular Data to Images**
- Automatically converts table rows to heatmap images
- Normalizes data to 0-255 range for visualization

✅ **Pre-built CNN Model**
- 2 Convolutional layers with max pooling
- Dense layers with dropout for regularization
- Binary classification output

✅ **Complete Pipeline**
- Data loading (CSV or synthetic)
- Train/test split (80/20)
- Model training with validation
- Performance evaluation
- Visual output and predictions

✅ **Easy to Extend**
- Modular code structure
- Configurable parameters
- Support for custom datasets

---

## 📁 Project Structure

```
heatmap-cnn-model/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore file
│
├── sample_data.csv        # Sample dataset for testing
│
├── outputs/               # Generated outputs (auto-created)
│   ├── heatmap_cnn_model.h5  # Trained model
│   ├── heatmap_samples.png   # Visual examples
│   └── training_history.png  # Training curves
│
└── docs/                  # Documentation (optional)
    └── DEPLOYMENT.md      # Deployment guide
```

---

## 📦 Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk**: ~2GB for dependencies

### Python Dependencies
```
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
tensorflow==2.13.0
Pillow==10.0.0
```

---

## 🚀 Installation

### Step 1: Clone or Download Repository

```bash
# If cloning from GitHub
git clone https://github.com/yourusername/heatmap-cnn-model.git
cd heatmap-cnn-model
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Installation Time**: ~5-10 minutes (depending on internet speed)

---

## 💻 Testing in VS Code

### Setup VS Code

1. **Install Python Extension**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Python" by Microsoft
   - Click Install

2. **Select Python Interpreter**
   - Open Command Palette (Ctrl+Shift+P)
   - Type "Python: Select Interpreter"
   - Choose the virtual environment you created

### Run the Application

#### Option 1: Using Terminal (Easiest)

```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run with synthetic data (sample data)
python main.py

# OR run with your CSV file
python main.py sample_data.csv
```

#### Option 2: Using VS Code Run Button

1. Open `main.py`
2. Click the ▶️ (Run) button in top right
3. Output appears in Terminal panel

#### Option 3: Debug Mode

1. Set breakpoints by clicking line numbers
2. Press F5 to start debugging
3. Step through code with F10/F11

### Monitor Execution

Watch the terminal output:
```
============================================================
HEATMAP + CNN MODEL - TABULAR TO VISION
============================================================

[STEP 1] Creating Sample Tabular Data...
✓ Data Shape: (200, 20)
✓ Features: 20
✓ Samples: 200
✓ Classes: 2

[STEP 2] Converting Tabular Data to Heatmap Images...
✓ Converted 200 heatmap images
...
```

---

## 📊 Usage

### Use with Synthetic Data (Built-in)

```bash
python main.py
```

This creates 200 sample records with 20 features.

### Use with Your CSV File

```bash
python main.py your_data.csv
```

**CSV Format Requirements:**
- Last column must be the target/label (0 or 1)
- All other columns are features
- Headers in first row

**Example CSV:**
```csv
Age,Income,Score,Purchases,Target
25,50000,85,100,0
35,75000,92,250,1
45,100000,95,500,1
```

### Use in Python Script

```python
from main import *

# Custom configuration
Config.DATA_SIZE = 500
Config.N_FEATURES = 30
Config.EPOCHS = 20

# Run pipeline
main(use_csv=True, csv_path='data.csv')
```

---

## 📈 Results & Output

After running, you'll get:

### 1. Terminal Output
```
Training Accuracy: 85.50%
Testing Accuracy: 82.30%
```

### 2. Output Files (in `outputs/` folder)

| File | Purpose |
|------|---------|
| `heatmap_cnn_model.h5` | Trained neural network model |
| `heatmap_samples.png` | 10 sample heatmap images with predictions |
| `training_history.png` | Loss and accuracy curves |

### 3. Visualizations

**Heatmap Examples:**
- Shows how tabular data converts to images
- Green labels = correct predictions
- Red labels = incorrect predictions

**Training History:**
- Left graph: Loss over 15 epochs
- Right graph: Accuracy over 15 epochs

---

## 🌐 GitHub Deployment

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Enter repository name: `heatmap-cnn-model`
3. Add description
4. Choose Public or Private
5. Click "Create repository"

### Step 2: Push Code to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Heatmap+CNN model"

# Add remote repository
git remote add origin https://github.com/yourusername/heatmap-cnn-model.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 3: Verify on GitHub

1. Go to https://github.com/yourusername/heatmap-cnn-model
2. Verify all files are there
3. Check README.md appears on main page

### Step 4: GitHub Actions (Optional - CI/CD)

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: python main.py
```

---

## 🔧 How It Works

### Pipeline Overview

```
Raw Tabular Data
       ↓
[STEP 1] Load & Understand
       ↓
[STEP 2] Convert to Heatmap Images
       ↓
[STEP 3] Split Train/Test (80/20)
       ↓
[STEP 4] Build CNN Model
       ↓
[STEP 5] Train on Images
       ↓
[STEP 6] Evaluate Performance
       ↓
[STEP 7] Make Predictions
       ↓
[STEP 8] Save Results & Visualizations
```

### Heatmap Conversion

```
Row: [5.85, -3.87, 7.58, ...]
     ↓
Normalize to 0-255
     ↓
Reshape to 28×28 grid
     ↓
Create heatmap image
```

### CNN Architecture

```
Input: 28×28 Image
    ↓
Conv2D (32 filters) → ReLU
    ↓
MaxPool (2×2)
    ↓
Conv2D (64 filters) → ReLU
    ↓
MaxPool (2×2)
    ↓
Flatten → 232,065 parameters
    ↓
Dense (128) → ReLU → Dropout
    ↓
Dense (64) → ReLU → Dropout
    ↓
Dense (1) → Sigmoid
    ↓
Output: 0 or 1
```

---

## 📚 Examples

### Example 1: Medical Data

```bash
# Predict disease from patient measurements
# CSV: Age, BloodPressure, Cholesterol, BloodSugar, Status
python main.py medical_data.csv
```

### Example 2: Financial Data

```bash
# Predict stock movement
# CSV: OpenPrice, ClosePrice, Volume, RSI, Target
python main.py stock_data.csv
```

### Example 3: E-commerce Data

```bash
# Predict product demand
# CSV: Price, Rating, Reviews, Sales, Category, Target
python main.py products_data.csv
```

---

## 🐛 Troubleshooting

### Issue 1: Module Not Found

```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```bash
pip install --upgrade tensorflow
```

### Issue 2: CUDA Errors

```
Could not load dynamic library 'libcuda.so.1'
```

**Solution:** (This is normal on CPU-only systems)
- Works fine without GPU
- Just slower training

### Issue 3: Memory Error

```
MemoryError: Unable to allocate memory
```

**Solution:**
```python
# Reduce batch size in main.py
Config.BATCH_SIZE = 8  # Was 16
```

### Issue 4: CSV Not Found

```
FileNotFoundError: your_data.csv
```

**Solution:**
```bash
# Make sure CSV is in same directory
ls sample_data.csv  # Check it exists

# Use full path if needed
python main.py /path/to/your_data.csv
```

---

## 👥 Contributing

Contributions welcome!

1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open Pull Request

---

## 📄 License

This project is licensed under MIT License - see LICENSE file for details.

---

## 📧 Support

- **Issues**: Open issue on GitHub
- **Questions**: Discussions tab
- **Email**: your-email@example.com

---

## 🔗 Links

- **GitHub Repository**: https://github.com/yourusername/heatmap-cnn-model
- **Documentation**: See docs/ folder
- **Dataset Examples**: sample_data.csv

---

## 🎯 Roadmap

- [ ] Add more model architectures (ResNet, VGG)
- [ ] Multi-class classification
- [ ] Model optimization & quantization
- [ ] Web API deployment (Flask/FastAPI)
- [ ] Docker support
- [ ] GUI application
- [ ] Jupyter notebook version

---

Last updated: 2024
Version: 1.0.0