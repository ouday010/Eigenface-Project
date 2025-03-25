# Eigenfaces Face Recognition System
A machine learning pipeline for face recognition using PCA (Eigenfaces) and SVM, implemented with scikit-learn.

📌 Key Features
- Implements classical Eigenfaces method using PCA
- Achieves **95%+ accuracy** on the Olivetti Faces dataset
- Includes hyperparameter tuning for optimal SVM performance
- Handles new face images via URL input
- Visualizes key components: eigenfaces, variance ratios, and confusion matrices
- Model persistence with Joblib for easy reuse

## � Dependencies
```bash
pip install numpy matplotlib scikit-learn opencv-python pillow requests joblib
