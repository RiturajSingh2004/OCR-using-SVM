# OCR Handwritten Digits Recognition using SVM

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/riturajsingh/ocr-digits/graphs/commit-activity)

A machine learning project that implements Optical Character Recognition (OCR) for handwritten digit classification using Support Vector Machine (SVM) algorithm. This project achieves high accuracy in recognizing handwritten digits from the popular digits dataset.

## 🚀 Features

- **High Accuracy**: Achieves excellent performance on digit recognition
- **SVM Implementation**: Uses Support Vector Machine with polynomial kernel
- **Data Visualization**: Displays sample digits and prediction results
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and confusion matrix
- **Model Persistence**: Saves trained model for future use
- **Professional Reporting**: Generates detailed performance reports

## 📊 Dataset

The project uses the built-in digits dataset from scikit-learn, which contains:
- **Samples**: 1,797 8x8 images of handwritten digits
- **Classes**: 10 digits (0-9)
- **Features**: 64 features per sample (8x8 pixel values)
- **Format**: Grayscale images with pixel values from 0-16

## 🛠️ Installation

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Dependencies

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib pickle-mixin
```

Or install from requirements.txt (if available):

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/RiturajSingh2004/OCR-using-SVM.git
cd OCR-using-SVM
```

## 🏃‍♂️ Usage

### Quick Start

Run the main script to execute the complete OCR pipeline:

```bash
python main.py
```

### What the Script Does

1. **Data Loading**: Loads the digits dataset
2. **Data Visualization**: Displays sample handwritten digits
3. **Data Preprocessing**: 
   - Splits data into training and testing sets (80/20)
   - Applies feature scaling using StandardScaler
4. **Model Training**: Trains SVM with polynomial kernel
5. **Prediction**: Makes predictions on test data
6. **Evaluation**: Comprehensive performance analysis
7. **Model Saving**: Saves the trained model as `OCR-SVM.pkl`

### Example Output

```
Dataset keys: dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
Data shape: (1797, 64)
Images shape: (1797, 8, 8)
Train set: (1437, 64) (1437,)
Test set: (360, 64) (360,)

Training SVM model...
Making predictions...
First 5 predictions: [0 1 2 3 4]

Evaluating model...
Avg F1-score: 0.9889
Accuracy: 0.9889
Precision: 0.9889
```

## 📈 Model Performance

The SVM model with polynomial kernel typically achieves:

- **Accuracy**: ~98.9%
- **F1-Score**: ~98.9%
- **Precision**: ~98.9%
- **Recall**: High across all digit classes

### Confusion Matrix

The model generates a detailed confusion matrix showing classification performance for each digit class.

## 🔧 Model Configuration

The SVM model uses the following hyperparameters:

- **Kernel**: Polynomial (`poly`)
- **Regularization (C)**: 10
- **Feature Scaling**: StandardScaler normalization

## 📁 Project Structure

```
ocr-handwritten-digits/
│
├── main.py                 # Main script with complete OCR pipeline
├── LICENSE                 # MIT License
├── README.md              # Project documentation
├── OCR-SVM.pkl            # Saved trained model (generated after running)
└── requirements.txt       # Python dependencies (optional)
```

## 🔍 Code Structure

### Main Functions

- `load_data()`: Loads the digits dataset
- `display_sample_digits()`: Visualizes sample digits
- `preprocess_data()`: Handles data preprocessing and splitting
- `train_svm_model()`: Trains the SVM classifier
- `make_predictions()`: Generates predictions on test data
- `evaluate_model()`: Comprehensive model evaluation
- `save_model()`: Saves the trained model
- `generate_report()`: Creates performance report

## 📊 Visualizations

The project includes several visualizations:

1. **Sample Digits**: Shows original handwritten digits from training data
2. **Predictions**: Displays test images with predicted labels
3. **Confusion Matrix**: Visual representation of classification performance

## 🤖 Using the Saved Model

After training, the model is saved as `OCR-SVM.pkl`. You can load and use it for new predictions:

```python
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model
with open('OCR-SVM.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions on new data
# (Remember to apply the same preprocessing as during training)
```

## 🔬 Technical Details

### Algorithm Choice

**Support Vector Machine (SVM)** was chosen because:
- Excellent performance on high-dimensional data
- Effective with limited training samples
- Robust against overfitting
- Good generalization capabilities for image classification

### Preprocessing Pipeline

1. **Data Splitting**: Stratified split to maintain class distribution
2. **Feature Scaling**: StandardScaler for normalization
3. **Array Conversion**: Ensures consistent data types

## 🚀 Future Improvements

- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Implementation of other algorithms (Random Forest, Neural Networks)
- [ ] Real-time digit recognition from camera input
- [ ] Web interface for digit recognition
- [ ] Support for custom handwritten digit images
- [ ] Cross-validation for robust performance estimation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Rituraj Singh**
- GitHub: [@RiturajSingh](https://github.com/RiturajSingh2004)

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for providing excellent machine learning tools
- The original creators of the digits dataset
- The open-source community for continuous inspiration

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [Digits Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html)

---

⭐ **Star this repository if you found it helpful!**
