🧠 MNIST Handwritten Digit Classification using TensorFlow

This project implements a Deep Learning model to classify handwritten digits (0–9) using the MNIST dataset with TensorFlow and Keras. It demonstrates the complete machine learning pipeline—from data loading and preprocessing to model training and evaluation.

📌 Project Overview

The MNIST dataset is a benchmark dataset in computer vision and deep learning. It consists of 70,000 grayscale images of handwritten digits:

60,000 training images

10,000 test images

Each image is 28 × 28 pixels.
The objective is to build a neural network that accurately predicts the digit shown in each image.

⚙️ Workflow

Load the MNIST dataset using TensorFlow

Visualize sample handwritten digits

Normalize pixel values (0–255 → 0–1)

One-hot encode labels (0–9)

Build a neural network using Keras Sequential API

Train the model

Evaluate model performance on test data

🏗️ Model Architecture

Input Layer: 28 × 28 images

Flatten Layer: Converts 2D image to 1D vector

Hidden Layer: Dense layer with 128 neurons and ReLU activation

Output Layer: Dense layer with 10 neurons and Softmax activation

📊 Results

Training Accuracy: ~99%

Test Accuracy: ~97–98%

The model generalizes well on unseen handwritten digits.

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Installation & Usage
git clone https://github.com/your-username/mnist-digit-classification.git
cd mnist-digit-classification
pip install tensorflow matplotlib numpy
python main.py


(You can also run this project in Jupyter Notebook or Google Colab.)
