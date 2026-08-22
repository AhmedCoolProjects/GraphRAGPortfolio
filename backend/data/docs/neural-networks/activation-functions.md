# Activation Functions

Activation functions are mathematical functions applied to the output of a neuron in a neural network. They introduce **non-linearity** into the model, allowing it to learn and represent complex patterns in the data. Without activation functions, a neural network would behave like a linear regression model, limiting its ability to capture intricate relationships.

## Heaviside Step Function

The simplest activation function which outputs either 0 or 1.

$$
f(x) = \begin{cases}
0 & \text{if } x < 0 \\
1 & \text{if } x \geq 0
\end{cases}
$$

## Rectified Linear Unit (ReLU)

ReLU is one of the most commonly used activation functions in deep learning. It outputs the input directly if it is positive; otherwise, it outputs zero.

$$
f(x) = \max(0, x)
$$

## Sigmoid Function

The Sigmoid function maps any input value to a value between 0 and 1, making it useful for binary classification problems.

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

## Softmax Function

The Softmax function is used in multi-class classification problems. It converts a vector of values into a probability distribution, where the sum of all probabilities equals 1.

$$
f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

## Python Implementation

```python title="activation_functions.py"
import numpy as np

class Activation_Functions:
    @staticmethod
    def heaviside(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0, 1)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))  # for numerical stability
        return exp_x / exp_x.sum(axis=1, keepdims=True) # axis=1 for row-wise
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        sig = Activation_Functions.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
```

:::info Derivative of Activation Functions

The derivatives of activation functions are crucial for the backpropagation process in training neural networks.

:::