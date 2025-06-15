#!/usr/bin/env python3
"""
OCR for Handwritten Digits Recognition using SVM
"""

# Import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn import datasets
import pickle
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data():
    """Load the digits dataset"""
    digits = datasets.load_digits()
    print("Dataset keys:", digits.keys())
    print("Data shape:", digits.data.shape)
    print("Images shape:", digits.images.shape)
    return digits

def display_sample_digits(digits):
    """Display first 4 digits from the dataset"""
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images[:4], digits.target[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()

def preprocess_data(digits):
    """Preprocess and split the data"""
    # Create feature and target arrays
    X = digits.data
    y = digits.target
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=365, stratify=y)
    
    # Convert to arrays
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    x_train = np.asarray(X_train)
    x_test = np.asarray(X_test)
    
    print('Train set:', x_train.shape, y_train.shape)
    print('Test set:', x_test.shape, y_test.shape)
    
    # Feature scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test, y_train, y_test, X_test, scaler

def train_svm_model(x_train, y_train):
    """Train SVM model"""
    svc = svm.SVC(C=10, kernel='poly')
    clf = svc.fit(x_train, y_train)
    return clf

def make_predictions(clf, x_test):
    """Make predictions on test data"""
    yhat = clf.predict(x_test)
    print("First 5 predictions:", yhat[:5])
    return yhat

def display_predictions(X_test, yhat):
    """Display first 4 test samples with predictions"""
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test[:4], yhat[:4]):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.show()

def evaluate_model(y_test, yhat):
    """Evaluate model performance"""
    svm_f1 = f1_score(y_test, yhat, average='weighted')
    svm_accuracy = accuracy_score(y_test, yhat)
    svm_precision = precision_score(y_test, yhat, average='micro')
    
    print("Avg F1-score: %.4f" % svm_f1)
    print("Accuracy: %.4f" % svm_accuracy)
    print("Precision: %.4f" % svm_precision)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, yhat, labels=np.unique(yhat)))
    
    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, yhat)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
    
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    
    return svm_f1, svm_accuracy, svm_precision

def save_model(clf, filename='OCR-SVM.pkl'):
    """Save the trained model"""
    pickle.dump(clf, open(filename, 'wb'))
    print(f"Model saved as {filename}")

def generate_report(svm_f1, svm_accuracy, svm_precision):
    """Generate performance report"""
    metric = {
        "Accuracy": [str(round(svm_f1*100, 2))+"%"],
        "F1 score": [str(round(svm_accuracy*100, 2))+"%"],
        "Precision": [str(round(svm_precision*100, 2))+"%"]
    }
    
    report = pd.DataFrame(metric)
    report = report.rename(index={0: 'Support Vector Machine'})
    print("\nPerformance Report:")
    print(report)
    return report

def main():
    """Main function to run the OCR pipeline"""
    print("Starting OCR for Handwritten Digits Recognition")
    print("=" * 50)
    
    # Load data
    digits = load_data()
    
    # Display sample digits
    display_sample_digits(digits)
    
    # Preprocess data
    x_train, x_test, y_train, y_test, X_test, scaler = preprocess_data(digits)
    
    # Train model
    print("\nTraining SVM model...")
    clf = train_svm_model(x_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    yhat = make_predictions(clf, x_test)
    
    # Display predictions
    display_predictions(X_test, yhat)
    
    # Evaluate model
    print("\nEvaluating model...")
    svm_f1, svm_accuracy, svm_precision = evaluate_model(y_test, yhat)
    
    # Save model
    save_model(clf)
    
    # Generate report
    report = generate_report(svm_f1, svm_accuracy, svm_precision)
    
    print("\nOCR pipeline completed successfully!")
    return clf, scaler, report

if __name__ == "__main__":
    model, scaler, performance_report = main()
