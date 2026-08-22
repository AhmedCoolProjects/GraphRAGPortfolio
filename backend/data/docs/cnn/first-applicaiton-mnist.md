# First Application on MNIST

In this section, we will build our first Convolutional Neural Network (CNN) to classify images from the MNIST dataset. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. We will use our classes created from scratch to implement the CNN architecture.

## Python Code

```python title="cnn_application_mnist.py"
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class NN:
    def __init__(self, s: List[int]):
        # ...
        pass
    def forward_pass(self, X: np.ndarray) -> (np.ndarray, List[tuple]):
        # ...
        pass
    def compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> float:
        # ...
        pass
    def backpropagation(self, AL: np.ndarray, Y: np.ndarray, cache: List[tuple]) -> dict:
        # ...
        pass
    def update_parameters(self, grads: dict, learning_rate: float = 0.001):
        # ...
        pass

class Conv_Layer:
    def __init__(self, in_channels, num_filters, filter_size, stride=1, padding=0):
        # ...
        pass
    def _zero_pad(self, X, pad):
        # ...
        pass
    def _relu(self, Z):
        # ...
        pass
    def _relu_derivative(self, Z):
        # ...
        pass
    def forward_pass(self, X_prev):
        # ...
        pass
    def backpropagation(self, dA):
        # ...
        pass
    def update_parameters(self, learning_rate=0.001):
        # ...
        pass

class Pool_layer:
    def __init__(self, filter_size=2, stride=2, mode='max'):
        # ...
        pass
    def _create_mask_from_window(self, x):
        # ...
        pass
    def forward_pass(self, A_prev):
        # ...
        pass
    def backpropagation(self, dA):
        # ...
        pass

class Flatten_Layer:
    def forward_pass(self, A_prev):
        # ...
        pass
    def backpropagation(self, dA):
        # ...
        pass


def load_and_prepare_data_cnn():
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # 1. Scale the data
    X = X / 255.0
    
    # 2. Reshape for CNN: (num_samples, 784) -> (num_samples, 28, 28, 1)
    X_reshaped = X.reshape(-1, 28, 28, 1)
    
    # 3. One-hot encode labels
    y_reshaped = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Y_onehot = encoder.fit_transform(y_reshaped)

    print("Data loaded and prepared.")

    # 4. Split the data
    X_train, X_test, Y_train_onehot, Y_test_onehot, _, y_test_orig = train_test_split(
        X_reshaped, Y_onehot, y.flatten(), test_size=0.1, random_state=42
    )

    # 5. Transpose Y to (classes, m)
    Y_train = Y_train_onehot.T
    Y_test = Y_test_onehot.T
    
    y_test_orig = y_test_orig.astype(int)

    return X_train, Y_train, X_test, Y_test, y_test_orig


class CNN:
    def __init__(self):
        print("Initializing CNN...")
        # Layer 1: Conv -> Pool
        self.conv1 = Conv_Layer(in_channels=1, num_filters=8, filter_size=3, stride=1, padding=0) # (m, 28, 28, 1) -> (m, 26, 26, 8)

        self.pool1 = Pool_Layer(filter_size=2, stride=2, mode='max') # (m, 26, 26, 8) -> (m, 13, 13, 8)
        # Layer 2: Flatten 
        self.flatten = Flatten_Layer() # (m, 13, 13, 8) -> (m, 1352)
        # Layer 3: Dense NN
        dense_input_size = 13*13*8
        self.dense_nn = NN(s=[dense_input_size, 64, 10]) # (1352, m) -> (10, m)
        self.layers = [self.conv1, self.pool1, self.flatten, self.dense_nn]
        print("CNN initialized.")
    
    def forward_pass(self, X):
        # Conv
        A_conv1, cache_conv1 = self.conv1.forward_pass(X)
        # Pool
        A_pool1, cache_pool1 = self.pool1.forward_pass(A_conv1)
        # Flatten
        A_flat, cache_flat = self.flatten.forward_pass(A_pool1)
        # Dense NN
        AL, cache_dense = self.dense_nn.forward_pass(A_flat)

        # store all caches for backpropagation
        caches = (
            cache_conv1,
            cache_pool1,
            cache_flat,
            cache_dense
        )

        return AL, caches
    
    def compute_cost(self, AL, Y):
        return self.dense_nn.compute_cost(AL, Y)
    
    def backpropagation(self, AL, Y, caches):
        (cache_conv1, cache_pool1, cache_flat, cache_dense) = caches
        # 4. Dense
        grads_dense, dA_flat = self.dense_nn.backpropagation(AL, Y, cache_dense)
        # 3. Flatten
        dA_pool1 = self.flatten.backpropagation(dA_flat)
        # 2. Pool
        dA_conv1 = self.pool1.backpropagation(dA_pool1, cache_pool1)
        # 1. Conv
        dA_prev = self.conv1.backpropagation(dA_conv1, cache_conv1)

        # Store dense grads
        self.dense_nn_grads = grads_dense

    def update_parameters(self, learning_rate=0.01):
        # Update Conv layer parameters
        self.conv1.update_parameters(learning_rate)
        # Update Dense NN parameters
        self.dense_nn.update_parameters(self.dense_nn_grads, learning_rate)

    def train(self, X, Y, epochs, learning_rate=0.01, batch_size=32):
        print(f"Starting training for {epochs} epochs...")
        costs = []
        m = X.shape[0]
        for epoch in range(epochs):
            epoch_cost = 0
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size, :, :, :]
                Y_batch = Y[:, i:i+batch_size]

                AL_batch, caches_batch = self.forward_pass(X_batch)
                cost = self.compute_cost(AL_batch, Y_batch)
                epoch_cost += cost * X_batch.shape[0]
                self.backpropagation(AL_batch, Y_batch, caches_batch)
                self.update_parameters(learning_rate)

            mean_epoch_cost = epoch_cost
            costs.append(mean_epoch_cost)

            print(f"Epoch {epoch+1}/{epochs}, Cost: {mean_epoch_cost:.4f}")

        print("Training completed.")
        return costs
    
    def predict(self, X):
        AL, _ = self.forward_pass(X)
        predictions = np.argmax(AL, axis=0)
        return predictions

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, y_test_orig = load_and_prepare_data_cnn()
    cnn_model = CNN()
    SUBSET_SIZE = 1000
    print("Training on subset of data...")
    X_train_subset = X_train[:SUBSET_SIZE]
    Y_train_subset = Y_train[:, :SUBSET_SIZE]
    costs = cnn_model.train(X_train_subset, Y_train_subset, epochs=25, learning_rate=0.01, batch_size=8)

    print("Evaluating on test set...")
    SUBSET_TEST_SIZE = 200
    X_test_subset = X_test[:SUBSET_TEST_SIZE]
    y_test_subset_orig = y_test_orig[:SUBSET_TEST_SIZE]
    predictions = cnn_model.predict(X_test_subset)
    accuracy = np.mean(predictions == y_test_subset_orig) * 100
    print(f"Test Set Accuracy: {accuracy:.2f}%")
```

## Output

```text
Training on subset of data...
Starting training for 25 epochs...
Epoch 1/25, Cost: 2324.9798
Epoch 2/25, Cost: 2261.9634
Epoch 3/25, Cost: 2189.5778
Epoch 4/25, Cost: 2037.9250
Epoch 5/25, Cost: 1783.1098
Epoch 6/25, Cost: 1486.0922
Epoch 7/25, Cost: 1232.4294
Epoch 8/25, Cost: 1042.2106
Epoch 9/25, Cost: 902.0981
Epoch 10/25, Cost: 796.7148
Epoch 11/25, Cost: 714.8593
Epoch 12/25, Cost: 649.2456
Epoch 13/25, Cost: 595.2510
Epoch 14/25, Cost: 549.9257
Epoch 15/25, Cost: 511.3004
Epoch 16/25, Cost: 477.9693
Epoch 17/25, Cost: 448.9134
Epoch 18/25, Cost: 423.3421
Epoch 19/25, Cost: 400.6551
Epoch 20/25, Cost: 380.3701
Epoch 21/25, Cost: 362.0987
Epoch 22/25, Cost: 345.5339
Epoch 23/25, Cost: 330.4288
Epoch 24/25, Cost: 316.5674
Epoch 25/25, Cost: 303.7871
Training completed.

Evaluating on test set...
Test Set Accuracy: 88.50%
```

## Notebook

Check This [CNN Application on MNIST Google Colab Notebook](https://gist.github.com/AhmedCoolProjects/8283bc04546d61057d5251199091351f) for an interactive experience!