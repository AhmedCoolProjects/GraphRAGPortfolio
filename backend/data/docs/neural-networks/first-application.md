# First Application (MNIST)

Let's take a multi-class classification problem as an example to illustrate how a neural network can be applied. Suppose we have a dataset of images of handwritten digits (0-9), and our goal is to classify each image into one of the 10 classes (digits).

### Step 1: Data Preparation

```python title="train_mnist.py"
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_and_prepare_data():
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    # 1. Flattening is already done in MNIST (28x28 images to 784 features)
    # 2. Scale the data (0-255 to 0-1)
    X = X / 255.0
    # 3. One-hot encode the labels
    y_reshaped = y.reshape(-1, 1) # One-hot encoder expects 2D array
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Y_onehot = encoder.fit_transform(y_reshaped)

    print("Data loaded and prepared.")

    # 4. Split the data
    # train_test_split expects (num_samples, num_features) or (num_samples, num_classes) for its inputs.
    # We pass X, Y_onehot, and y (original labels) to split them consistently.
    X_train_raw, X_test_raw, Y_train_onehot_raw, Y_test_onehot_raw, _, y_test_orig_raw = train_test_split(X, Y_onehot, y.flatten(), test_size=0.1, random_state=42)

    # Transpose the data to match the expected format (features, samples) for X 
    # and (classes, samples) for Y, as implied by subsequent usage like X_test[:, image_index].
    X_train = X_train_raw.T
    X_test = X_test_raw.T
    Y_train = Y_train_onehot_raw.T
    Y_test = Y_test_onehot_raw.T

    # Convert all to int
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    Y_train = Y_train.astype(float)
    Y_test = Y_test.astype(float)
    y_test_orig_raw = y_test_orig_raw.astype(int)

    # y_test_orig_raw is already 1D (num_samples,) from y.flatten() and train_test_split
    return X_train, X_test, Y_train, Y_test, y_test_orig_raw

```


### Step 2: Define the Neural Network Architecture

```python title="nn.py"
from typing import List, Tuple

import numpy as np

np.random.seed(42)

class NN:
    def __init__(self, s: List[int]):
        '''
        s: List of layer sizes, where s[0] is the input layer size, s[1] is the first hidden layer size, ..., s[L] is the output layer size
        '''
        self.s = s
        self.L = len(s) - 1
        # Xavier (Glorot) initialization for weights
        self.weights = [np.random.randn(s[l], s[l-1]) * np.sqrt(1 / s[l-1]) for l in range(1, self.L + 1)]
        self.biases = [np.zeros((s[l], 1)) for l in range(1, self.L + 1)]
        self.f = Activation_Functions.sigmoid
        self.f_output = Activation_Functions.softmax
        self.cost = Cost_Functions.ce
        self.cost_grad = Cost_Functions.ce_grad

    def forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, List[tuple]]:
        # ...
        pass

    def compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> float:
        # ...
        pass

    def backpropagation(self, AL: np.ndarray, Y: np.ndarray, cache: List[tuple]) -> dict:
        # ...
        pass

        return grads

    def update_parameters(self, grads: dict, learning_rate: float = 0.001):
        # ...
        pass

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100, learning_rate: float = 0.001, batch_size: int = 20, print_cost: bool = True) -> None:
        '''
        The main training loop with mini-batch gradient descent.
        '''
        costs = []
        m = X.shape[1] # total number of samples

        for epoch in range(epochs):
            # Shuffle the training data at the beginning of each epoch
            permutation = np.random.permutation(m)
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation]

            # Iterate through mini-batches
            for i in range(0, m, batch_size):
                X_batch = shuffled_X[:, i:i + batch_size]
                Y_batch = shuffled_Y[:, i:i + batch_size]

                # 1. Forward Pass for the batch
                AL_batch, cache_batch = self.forward_pass(X_batch)
                # 2. Backpropagation for the batch
                grads_batch = self.backpropagation(AL_batch, Y_batch, cache_batch)
                # 3. Update Parameters using batch gradients
                self.update_parameters(grads_batch, learning_rate)

            # Calculate and print the overall cost after all batches for the epoch
            AL_full, _ = self.forward_pass(X) # Forward pass on full dataset for cost calculation
            cost = self.compute_cost(AL_full, Y)

            if print_cost and epoch % (epochs // 10 if epochs >= 10 else 1) == 0:
                print(f"Cost after epoch {epoch}: {cost:.4f}")
            if epoch % 10 == 0:
                costs.append(cost)
        return costs

    def predict(self, X: np.ndarray) -> np.ndarray:
        AL, _ = self.forward_pass(X)
        predictions = np.argmax(AL, axis=0)
        return predictions
```

### Step 3: Train the Neural Network

```python title="train_mnist.py"
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from nn import NN

def load_and_prepare_data():
    # ...
    pass

if __name__ == "__main__":
    # 1. Get Data
    X_train, X_test, Y_train, Y_test, y_test_orig = load_and_prepare_data()

    # 2. Define Network Architecture:
    # s[0] = 784 (input layer)
    # s[1] = 64 (hidden layer)
    # s[2] = 32 (hidden layer)
    # s[3] = 10 (output layer)
    s = [784, 64, 32, 10]

    print(f"Creating Neural Network with architecture: {s}")
    
    model = NN(s)

    # 3. Train the Model
    print("Starting training...")
    costs = model.train(X_train, Y_train, epochs=200, learning_rate=0.01)
    print("Training completed.")

    # 4. Evaluate the Model
    print("Evaluating the model on test data...")
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test_orig)

    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
```

### Output

```text
Starting training...
Cost after epoch 0: 2.0526
Cost after epoch 20: 0.2197
Cost after epoch 40: 0.1341
Cost after epoch 60: 0.0969
Cost after epoch 80: 0.0750
Cost after epoch 100: 0.0599
Cost after epoch 120: 0.0491
Cost after epoch 140: 0.0404
Cost after epoch 160: 0.0335
Cost after epoch 180: 0.0284
Training completed.

Evaluating the model on test data...
Test Set Accuracy: 97.11%
```

### Notebook

Check This [NN Application on MNIST Google Colab Notebook](https://gist.github.com/AhmedCoolProjects/c6546effe88d08b9ddc04ed587fbdf20) for an interactive experience!