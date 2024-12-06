# Deep-learning--Feed-Forward-Neural-Network
1. Dataset: MNIST

Dataset Overview:

MNIST consists of grayscale images of handwritten digits (0–9), each of size 28x28 pixels, and their corresponding labels. The dataset is a standard benchmark in machine learning, especially for image classification tasks.

Preprocessing:

Images are normalized to scale pixel values between 0 and 1. The images are flattened into 1D vectors of size 784 (28x28). The dataset is split into training and test sets, with loaders created for mini-batch processing during training and evaluation.

2. Model Architecture

Model Type:

The notebook implements a Feedforward Neural Network (FNN).

Network Structure:

Input Layer: Accepts 784 features (flattened 28x28 images).

Hidden Layer:

A fully connected layer with a specified number of neurons. Activation function applied (likely ReLU) to introduce non-linearity.

Output Layer:

A fully connected layer with 10 output neurons (one for each digit class). Softmax activation is likely used to output class probabilities.

3. Training

Loss Function: The loss function used is likely CrossEntropyLoss, which is standard for multi-class classification problems. Optimizer:

A gradient-based optimization algorithm such as SGD or Adam is used to update the model weights. Training Process:

The training loop iterates through the dataset for a fixed number of epochs. For each mini-batch:

Forward pass: Compute predictions using the model.

Loss calculation: Compare predictions with true labels using the loss function.

Backward pass: Compute gradients via backpropagation.

Parameter updates: Update model weights using the optimizer.

4. Evaluation

Accuracy Measurement: The trained model is evaluated on the test dataset to determine its classification performance. Final Accuracy Achieved: The model achieves 96% accuracy on the test dataset, indicating its strong performance in recognizing handwritten digits.
