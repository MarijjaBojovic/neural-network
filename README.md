# 🍎 Fruit Classification Suite: CNN & Transfer Learning

This project implements an end-to-end Deep Learning pipeline for classifying fruit images (Apples, Oranges, and Bananas). It explores two distinct architectural approaches: a **Custom CNN** built from scratch and **Transfer Learning** using the MobileNetV2 architecture.

---

## 🚀 Key Features

### 1. 🖼️ Data Preprocessing & Augmentation
* **Automated Pipeline:** Loads, resizes, and shuffles images across train, validation, and test splits.
* **Class Balancing:** Utilizes `compute_class_weight` to handle potential dataset imbalances during training.
* **Data Augmentation:** Implements a Keras-native augmentation layer including:
    * Random Horizontal & Vertical Flips
    * Random Rotations (0.25 factor)
    * Random Zooming

### 2. 🏗️ Model Architectures
* **Custom CNN:** * 3x Convolutional layers with increasing filters (16, 32, 64).
    * **Batch Normalization** for faster convergence and stability.
    * **Max Pooling** and **Dropout (0.3)** to prevent overfitting.
* **Transfer Learning (MobileNetV2):**
    * Uses a pre-trained **MobileNetV2** base (ImageNet weights).
    * Implements Feature Extraction by freezing the base model and adding custom Dense/GlobalAveragePooling layers on top.

### 3. 🧪 Training & Evaluation
* **Early Stopping:** Monitors `val_accuracy` to restore the best weights and prevent overtraining.
* **Normalization:** Integrated `preprocess_input` for ConvNeXt/MobileNet compatibility.
* **Visualization:** Generates detailed learning curves (Accuracy/Loss) and Confusion Matrices.

---

## 🛠️ Technical Stack
* **Framework:** Keras / TensorFlow
* **Libraries:** * `NumPy` & `PIL` (Image processing)
    * `Matplotlib` (Data visualization)
    * `Scikit-learn` (Metrics: Accuracy, Classification Report, Confusion Matrix)
* **Architecture Models:** Custom CNN & MobileNetV2

---

## 📈 Results & Visualizations
The suite provides several diagnostic tools to verify model health:
* **Distribution Histograms:** Checks the balance of the dataset splits.
* **Augmentation Samples:** Visualizes how the model "sees" modified training images.
* **Prediction Analysis:** Dedicated plots for **Correctly Classified** vs. **Misclassified** images to identify edge cases (e.g., lighting or fruit positioning).
* **Confusion Matrices:** Normalized displays to evaluate precision and recall across all three classes.

---

## 📂 Project Structure
```text
├── fruits/
│   ├── train/          # Apple, Orange, Banana subfolders
│   ├── validation/     # Apple, Orange, Banana subfolders
│   └── test/           # Apple, Orange, Banana subfolders
├── fruit_classifier.py # Main implementation script
└── README.md
