# Forward Pass

In a neural network, the **Forward Pass** is the initial step where input data is passed through the network to generate an output. This process involves computing the weighted sum of inputs, adding biases, and applying activation functions at each neuron in the network.

## Steps in Forward Pass

1. **Input Layer**: The input data is fed into the network through the input layer. Each feature of the input data corresponds to a neuron in this layer. So the number of neurons in the input layer equals the number of features in the input data.
2. **Hidden Layers**: The data is then passed to one or more hidden layers. Each neuron in a hidden layer performs the following operations:
   - Computes the weighted sum of its inputs.
   - Adds a bias term.
   - Applies an activation function to introduce non-linearity. _(See [Activation Functions](./activation-functions) for more details.)_

   Mathematically, for a neuron \( j \) in layer \( l \):
   $$
   z_j^{(l)} = \sum_{i} w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}
   $$
   $$
   a_j^{(l)} = f(z_j^{(l)})
   $$
   Where:
   - $ w_{ij}^{(l)} $ is the weight from neuron \( i \) in layer \( l-1 \) to neuron \( j \) in layer \( l \).
   - $ a_i^{(l-1)} $ is the activation of neuron \( i \) in layer \( l-1 \).
   - $ b_j^{(l)} $ is the bias term for neuron \( j \) in layer \( l \).
   - $ f $ is the activation function.
3. **Output Layer**: Finally, the data reaches the output layer, where the final predictions are made. The output layer may use different activation functions depending on the type of problem (e.g., Softmax for multi-class classification, Sigmoid for binary classification).

After the forward pass, the output of the network is compared to the actual target values using a **Cost Function** to measure the error. _(See [Cost Functions](./cost-functions) for more details.)_

:::tip Dimensionality and Notation
- $ l $: Index of the current layer (0 for input, 1 for first hidden layer, ..., L for output layer).
- $ L $: Index of last layer (output layer).
- $ m $: Number of training examples in the batch.
- $ s[l] $: Number of neurons in layer $ l $.
- $ f $: Activation function applied element-wise.
---
- $$ X $$: Input data, a batch of $m$ examples each with $n$ features. Shape: **(n, m)**.
- $ w_{ij}^{(l)} $: The weight connecting the neuron $i$ in layer $l$ to neuron $j$ in layer $l-1$.
- $ W^{(l)} $: Weight matrix for layer $l$. Shape: $ (s[l], s[l-1]) $.
- $ b_i^{(l)} $: Bias for neuron $i$ in layer $l$.
- $ b^{(l)} $: Bias vector for layer $l$. Shape: $ (s[l], 1) $.
- $ z_i^{(l)} $: Weighted input to neuron $i$ in layer $l$.
- $ Z^{(l)} $: Weighted input matrix for layer $l$. Shape: $ (s[l], m) $.
- $ a_i^{(l)} $: Activation (output) of neuron $i$ in layer $l$.
- $ A^{(l)} $: Activation matrix for layer $l$. Shape: $ (s[l], m) $.

:::



## Python Implementation

```python title="neural_netwrok.py"
import numpy as np
from typing import List, Callable
from activation_functions import Activation_Functions
from cost_functions import Cost_Functions

np.random.seed(42)

class NN:
    def __init__(self, s: List[int]):
        '''
        s: List of layer sizes, where s[0] is the input layer size, s[1] is the first hidden layer size, ..., s[L] is the output layer size
        '''
        self.s = s
        self.L = len(s) - 1 
        self.weights = [np.random.randn(s[l], s[l-1]) * 0.01 for l in range(1, self.L + 1)]
        self.biases = [np.zeros((s[l], 1)) for l in range(1, self.L + 1)]
        self.f = Activation_Functions.sigmoid
        self.f_output = Activation_Functions.softmax
        self.cost = Cost_Functions.ce
        self.cost_grad = Cost_Functions.ce_grad

    def forward_pass(self, X: np.ndarray) -> (np.ndarray, List[tuple]):
        '''
        X: Input data of shape (n, m)
        '''
        cache = [] # only hidden and output layers
        A = X

        for l in range(self.L):
            A_prev = A

            W = self.weights[l]
            b = self.biases[l]
            # Linear Step
            Z = np.dot(W, A_prev) + b
            # Activation Step
            if l == self.L - 1:
                A = self.f_output(Z)
            else:
                A = self.f(Z)
            
            cache.append((A_prev, W, b, Z))

        return A, cache

    def compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> float:
        cost = self.cost(Y, AL)
        return cost
```