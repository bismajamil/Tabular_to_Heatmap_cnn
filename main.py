#!/usr/bin/env python3
"""
Heatmap + CNN Model - ULTRA-IMPROVED VERSION
Convert Tabular Data to Heatmap Images and train CNN Model
Author: Data Science Team
Version: 3.0.0
OPTIMIZATIONS: Maximum Accuracy, Better Preprocessing, Advanced Techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
class Config:
    """Configuration for the model"""
    DATA_SIZE = 200
    N_FEATURES = 20
    TEST_SIZE = 0.2
    EPOCHS = 50  # ✅ Increased to 50 (with early stopping)
    BATCH_SIZE = 4  # ✅ Reduced to 4 (better learning)
    IMAGE_SIZE = 32  # ✅ Increased to 32 (more information)
    RANDOM_STATE = 42
    
    # ✅ Use absolute path so files are saved where you can find them!
    OUTPUT_DIR = r'C:\Users\dell\OneDrive\Documents\Data classification using AI\outputs'
    
    @classmethod
    def setup(cls):
        """Create output directory if not exists"""
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Output directory: {cls.OUTPUT_DIR}")


# ============================================
# UTILITY FUNCTIONS
# ============================================
def create_sample_data(n_samples=200, n_features=20):
    """Create synthetic tabular data"""
    print("\n" + "="*60)
    print("[STEP 1] Creating Sample Tabular Data")
    print("="*60)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),  # ✅ More informative features
        n_redundant=int(n_features * 0.2),
        n_clusters_per_class=2,  # ✅ Multiple clusters
        n_classes=2,
        random_state=Config.RANDOM_STATE
    )
    
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    print(f"✓ Data Shape: {X.shape}")
    print(f"✓ Features: {n_features}")
    print(f"✓ Samples: {n_samples}")
    print(f"✓ Classes: 2")
    
    return X, y, feature_names


def load_data_from_csv(filepath):
    """Load data from CSV and apply advanced preprocessing"""
    print(f"\n✓ Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()
    
    # ✅ Convert text labels to numbers
    if isinstance(y[0], str):
        print(f"✓ Converting text labels to numbers...")
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"✓ Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    
    # ✅ IMPROVEMENT: Advanced preprocessing
    print(f"✓ Applying advanced preprocessing...")
    
    # Remove outliers using IQR method
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
    X = X[mask]
    y = y[mask]
    print(f"✓ Removed outliers. Remaining samples: {len(X)}")
    
    # ✅ Advanced normalization (0-1 range for better heatmaps)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    print(f"✓ Features normalized to 0-1 range (MinMaxScaler)")
    
    print(f"✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Classes: {len(np.unique(y))}")
    
    return X, y, feature_names


def table_to_heatmap_image(data_row, size=32):
    """
    ✅ IMPROVED: Better heatmap conversion with padding
    """
    # Already normalized to 0-1, convert to 0-255
    if isinstance(data_row, np.ndarray):
        normalized = data_row * 255
    else:
        normalized = data_row
    
    total_features = len(normalized)
    target_size = size * size
    
    if total_features < target_size:
        # ✅ IMPROVEMENT: Smart padding
        padded = np.full(target_size, 128, dtype=np.float32)
        
        # Place features in center
        start_idx = (target_size - total_features) // 2
        padded[start_idx:start_idx + total_features] = normalized
        heatmap = padded.reshape(size, size)
    else:
        # ✅ Take first 'target_size' features
        heatmap = normalized[:target_size].reshape(size, size)
    
    return heatmap.astype(np.uint8)


def convert_data_to_heatmaps(X):
    """Convert tabular data to heatmap images"""
    print("\n" + "="*60)
    print("[STEP 2] Converting Tabular Data to Heatmap Images")
    print("="*60)
    
    X_images = np.array([table_to_heatmap_image(row, Config.IMAGE_SIZE) for row in X])
    X_images = X_images / 255.0  # Normalize to 0-1
    
    print(f"✓ Converted {len(X_images)} images")
    print(f"✓ Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE} pixels")
    
    return X_images


def split_data(X_images, y):
    """Split data with stratification"""
    print("\n" + "="*60)
    print("[STEP 3] Splitting Data (Train/Test)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_images, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    X_train = X_train.reshape(-1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1)
    X_test = X_test.reshape(-1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1)
    
    print(f"✓ Training set: {X_train.shape[0]} images")
    print(f"✓ Testing set: {X_test.shape[0]} images")
    print(f"✓ Image shape: ({Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}, 1)")
    
    return X_train, X_test, y_train, y_test


def build_cnn_model(num_classes=2):
    """
    ✅ IMPROVED: More sophisticated CNN architecture
    """
    print("\n" + "="*60)
    print("[STEP 4] Building CNN Model")
    print("="*60)
    
    if num_classes == 2:
        # Binary classification with improved architecture
        model = keras.Sequential([
            # ✅ Block 1: More filters
            layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                         input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # ✅ Block 2: More filters
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # ✅ Block 3: More filters
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # ✅ Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output
            layers.Dense(1, activation='sigmoid')
        ])
        
        # ✅ Better optimizer with learning rate decay
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # Multi-class classification
        model = keras.Sequential([
            layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                         input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    print("✓ Model built successfully")
    model.summary()
    
    return model


def train_model(model, X_train, y_train):
    """
    ✅ IMPROVED: Advanced training with callbacks
    """
    print("\n" + "="*60)
    print("[STEP 5] Training CNN Model")
    print("="*60)
    
    # ✅ Data augmentation for better generalization
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    
    # ✅ Callbacks for better training
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
        epochs=Config.EPOCHS,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        steps_per_epoch=len(X_train) // Config.BATCH_SIZE
    )
    
    print("✓ Training completed")
    
    return history


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model"""
    print("\n" + "="*60)
    print("[STEP 6] Evaluating Model")
    print("="*60)
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    metrics = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    
    print(f"✓ Training Accuracy: {train_acc*100:.2f}%")
    print(f"✓ Testing Accuracy: {test_acc*100:.2f}%")
    
    return metrics


def make_predictions(model, X_test, y_test, n_samples=5):
    """Make predictions"""
    print("\n" + "="*60)
    print("[STEP 7] Making Predictions")
    print("="*60)
    
    predictions = model.predict(X_test[:n_samples], verbose=0)
    
    print(f"\nSample Predictions (First {n_samples} test samples):")
    print("-" * 50)
    
    correct = 0
    for i, (pred, actual) in enumerate(zip(predictions[:n_samples], y_test[:n_samples])):
        if pred.shape[0] == 1:
            pred_class = 1 if pred[0] > 0.5 else 0
            confidence = pred[0] if pred[0] > 0.5 else (1 - pred[0])
        else:
            pred_class = np.argmax(pred)
            confidence = np.max(pred)
        
        is_correct = pred_class == actual
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i+1}: Predicted={pred_class}, Actual={actual}, Confidence={confidence*100:.2f}%")
    
    print(f"\nCorrect Predictions: {correct}/{n_samples} ({correct/n_samples*100:.1f}%)")


def save_visualizations(history, X_test, y_test, model):
    """Save visualizations"""
    print("\n" + "="*60)
    print("[STEP 8] Creating Visualizations")
    print("="*60)
    
    # Plot 1: Heatmap samples
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle('Heatmap Images with Predictions', fontsize=14, fontweight='bold')
    
    predictions = model.predict(X_test[:10], verbose=0)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(X_test[:10]):
            ax.imshow(X_test[idx].reshape(Config.IMAGE_SIZE, Config.IMAGE_SIZE), cmap='hot', interpolation='nearest')
            
            if predictions[idx].shape[0] == 1:
                pred_class = 1 if predictions[idx][0] > 0.5 else 0
                conf = max(predictions[idx][0], 1-predictions[idx][0])
            else:
                pred_class = np.argmax(predictions[idx])
                conf = np.max(predictions[idx])
            
            color = 'green' if pred_class == y_test[idx] else 'red'
            ax.set_title(f'Pred: {pred_class} (Conf: {conf*100:.0f}%)', 
                        color=color, fontsize=9, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(Config.OUTPUT_DIR, 'heatmap_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Plot 2: Training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o')
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Model Loss Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Model Accuracy Over Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(Config.OUTPUT_DIR, 'training_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def save_model(model):
    """Save trained model"""
    output_path = os.path.join(Config.OUTPUT_DIR, 'heatmap_cnn_model.h5')
    model.save(output_path)
    print(f"✓ Saved: {output_path}")


def print_summary(metrics, num_samples, num_features, num_classes, data_source="Synthetic"):
    """Print summary"""
    print("\n" + "="*60)
    print("✓ PROCESS COMPLETE!")
    print("="*60)
    print("\n📊 SUMMARY:")
    print(f"  • Data Source: {data_source}")
    print(f"  • Tabular Data: {num_samples} records × {num_features} features")
    print(f"  • Heatmap Images: {num_samples} images ({Config.IMAGE_SIZE}×{Config.IMAGE_SIZE})")
    print(f"  • Model Type: Advanced CNN (Convolutional Neural Network)")
    print(f"  • Number of Classes: {num_classes}")
    print(f"  • Training Accuracy: {metrics['train_acc']*100:.2f}%")
    print(f"  • Testing Accuracy: {metrics['test_acc']*100:.2f}%")
    print(f"\n📁 OUTPUT FILES SAVED TO:")
    print(f"  📂 {Config.OUTPUT_DIR}")
    print(f"\n  Files created:")
    print(f"  1. heatmap_samples.png")
    print(f"  2. training_history.png")
    print(f"  3. heatmap_cnn_model.h5")
    print("\n✓ Open File Explorer and navigate to the folder above to view files!")
    print("="*60 + "\n")


# ============================================
# MAIN EXECUTION
# ============================================
def main(use_csv=False, csv_path=None):
    """Main execution function"""
    Config.setup()
    
    print("\n" + "="*60)
    print("HEATMAP + CNN MODEL - TABULAR TO VISION")
    print("ULTRA-IMPROVED VERSION v3.0 (MAXIMUM ACCURACY)")
    print("="*60)
    
    # ✅ Auto-load iris.csv if it exists
    if not use_csv and not csv_path:
        iris_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris.csv')
        if os.path.exists(iris_path):
            print(f"\n🔧 Found iris.csv! Using real data for best results...")
            use_csv = True
            csv_path = iris_path
    
    # Load Data
    if use_csv and csv_path:
        X, y, feature_names = load_data_from_csv(csv_path)
        data_source = "Iris Dataset (Real Data)"
    else:
        X, y, feature_names = create_sample_data(Config.DATA_SIZE, Config.N_FEATURES)
        data_source = "Synthetic Data"
    
    num_classes = len(np.unique(y))
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    print(f"✓ Number of classes detected: {num_classes}")
    
    # Convert to Heatmaps
    X_images = convert_data_to_heatmaps(X)
    
    # Split Data
    X_train, X_test, y_train, y_test = split_data(X_images, y)
    
    # Build Model
    model = build_cnn_model(num_classes=num_classes)
    
    # Train Model
    history = train_model(model, X_train, y_train)
    
    # Evaluate Model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Make Predictions
    make_predictions(model, X_test, y_test)
    
    # Save Visualizations
    save_visualizations(history, X_test, y_test, model)
    
    # Save Model
    save_model(model)
    
    # Print Summary
    print_summary(metrics, num_samples, num_features, num_classes, data_source)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"\n🔧 Using CSV file: {csv_file}")
        main(use_csv=True, csv_path=csv_file)
    else:
        print("\n🔧 Auto-detecting data source...")
        main(use_csv=False)