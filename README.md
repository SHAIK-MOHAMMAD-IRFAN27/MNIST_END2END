# Handwritten Digit Classifier

A deep learning project that uses Convolutional Neural Networks (CNN) and TensorFlow to classify handwritten digits from the MNIST dataset.

## Features

- Built with TensorFlow and Streamlit
- Uses CNN architecture for better accuracy
- Interactive web interface for digit prediction
- Supports upload of handwritten digit images (28x28 pixels)
- Real-time predictions

## Requirements

- Python 3.x
- TensorFlow
- Streamlit
- NumPy
- Pillow (PIL)

## Usage

1. Make sure you have all the required dependencies installed
2. Run the Streamlit app:
   ```
   streamlit run stlformodel.py
   ```
3. Upload a 28x28 pixel grayscale image of a handwritten digit
4. Click "Predict" to see the model's prediction

## Model Architecture

The model uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to achieve high accuracy in digit recognition.

## Files

- `stlformodel.py`: Main Streamlit application file
- `mnist_cnn_model.h5`: Trained model file (not included in repository) 