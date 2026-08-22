# RNN

**NN** and **CNN** are **feed-forward**: the data flows in one direction, from input to output. They are great for data where order doesn't matter, like classifying a single image.

**RNN** _(Recurrent Neural Networks)_ are built for **sequential data**, where the order of the data points matters, such as:
- Time series data _(e.g., stock prices over time)_
- Text data _(e.g., sentences in a paragraph)_
- Speech data _(e.g., audio signals)_
  
The **Recurrent** part means the network has a **loop**. It processes one item in the sequence _(e.g., one word)_, and the output of that step is **fed back** into the network as an input for the next step.

This loop gives RNNs a form of **memory**, allowing it to keep track of what it has seen before.

## How RNNs Work

Let's say we have a sequence of words: "The cat sat on the mat." An RNN would process this sequence one word at a time:
1. It takes the first word "The" and processes it, producing an output and a hidden state. _(the hidden state is like the memory of what the RNN has seen so far)_
2. It then takes the next word "cat" along with the hidden state from the previous step, processes them together, and produces a new output and updated hidden state.
3. This continues for each word in the sequence.

## Foward Pass

At each time step $t$, the RNN takes two inputs:
- The current input $x_t$ (e.g., the current word in the sequence)
- The previous hidden state $h_{t-1}$ (the memory from the last time step)


### Hidden State

To create the new memory $h_t$, the network combines its two inputs $x_t$ and $h_{t-1}$.

Since these are two different sources of information, it uses two different sets of weights to process them:

- $W_{xh}$: weights for the current input $x_t$
- $W_{hh}$: weights for the previous hidden state $h_{t-1}$

The math is simple "weighted sum" just like our Dense NN, but with two parts added together:
1. **Process Input**: $W_{xh} \cdot x_t$
2. **Process Memory**: $W_{hh} \cdot h_{t-1}$
3. **Combine**: Add them together with a bias $b_h$
4. **Activation**: Pass the result through an activation function. In classic RNNs, we often use the hyperbolic tangent function **tanh**.

$$
h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)
$$

<details>
<summary> Why **tanh**? </summary>

It's just like the *sigmoid* function, but it squashes values to be between **-1 and 1** instead of 0 and 1. This is often better for hidden states, as it allows the network to **strengthen** or **weaken** a memory in either a positive or negative direction.

</details>

### Output

This is a separate, simpler step. To make a prediction $\hat{y}_t$, the network only uses the current hidden state $h_t$ we just calculated.

This is a standard feed-forward step, just like in a Dense NN:
1. Take the new hidden state $h_t$
2. Multiply it by a new set of **output weights** $W_{hy}$
3. Add a bias $b_y$
4. Pass it through a final activation function _(like softmax if we're predicting the next word or character)_

The equation for the final prediction is:

$$
\hat{y}_t = \text{softmax}(W_{hy} \cdot h_t + b_y)
$$

### Summary

The key is that the same sets of weights _($W_{xh}$, $W_{hh}$, and $W_{hy}$)_ and biases _($b_h$ and $b_y$)_ are used at every time step. This allows the RNN to generalize across different positions in the sequence. 

## Backpropagation

Before we dive into backpropagation through time _(BPTT)_, let's take a look on how the cost is computed for RNNs.

### Cost Function

The cost function for NNs and CNNs is computed at the very end of the network after a single forward pass. However, in RNNs, since we have multiple time steps, we typically compute the cost at each time step _(called **Local Cost** $C_t$)_ and then sum them up to get the **Total Cost** $C$ for the entire sequence.

A local cost $C_t$ if we have the Cross-Entropy loss for classification tasks would be:

$$
C_t = - \sum_{i} y_{t,i} \log(\hat{y}_{t,i})
$$

Where:
- $y_{t,i}$ is the true label (one-hot encoded) for class $i$ at time step $t$.
- $\hat{y}_{t,i}$ is the predicted probability for class $i$ at time step $t$.

and Thus,

$$
C = \sum_{t=1}^{T} C_t
$$

Where:
- $T$ is the total number of time steps in the sequence.
- $C_t$ is the cost at time step $t$, which can be computed using a loss function like cross-entropy or mean squared error, depending on the task.

### BPTT

This is the "learning" part. Because the network is **unrolled** through time _(that is, we visualize each time step as a separate layer in a deep network)_, we have to propagate the erro backward through time. This is called **Backpropagation Through Time** (BPTT).

Here's how it works:
1. **Start at the End**: We begin by calculating the gradient of the cost at the final time step $T$ with respect to the output $\hat{y}_T$.
2. **Move Backwards**: The gradient at $t=T$ flows to the $t=T-1$ cell. The total gradient at $t=T-1$ is now a combination of two things:
    - The gradient from its own output *$\hat{y}_{T-1}$*
    - The gradient flowing back from the next time step *$t=T$* through the hidden state *$h_{T-1}$*

    $$
    \frac{\partial C}{\partial h_{T-1}} = \frac{\partial C_{T-1}}{\partial h_{T-1}} + \frac{\partial C_{T}}{\partial h_{T}} \cdot \frac{\partial h_{T}}{\partial h_{T-1}}
    $$

    The gradient of the $\hat{y}_{T}$ is:

    $$
    \frac{\partial C_{T}}{\partial \hat{y}_{T}} = \hat{y}_{T} - y_{T}
    $$

    <details>

    <summary> Explanation </summary>

    We have: $C_T = - \sum_{i} y_{T,i} \log(\hat{y}_{T,i})$. Taking the derivative with respect to $\hat{y}_{T,i}$ gives us:

    $$
    \frac{\partial C_{T}}{\partial \hat{y}_{T,i}} = - \frac{y_{T,i}}{\hat{y}_{T,i}}
    $$

    However, when we consider the softmax output layer, the derivative simplifies to:

    $$
    \frac{\partial C_{T}}{\partial \hat{y}_{T}} = \hat{y}_{T} - y_{T}
    $$

    </details>

3. **Repeat**: This process continues backward through all time steps until we reach the first time step $t=1$.

The most important part is how the *shared weights* _($W_{xh}$, $W_{hh}$, and $W_{hy}$)_ get updated.
- The gradient for $W_{hy}$ _(the output weights)_ is just the sum of the local output gradients from each time step.

    $$
    \frac{\partial C}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial C_t}{\partial W_{hy}}
    $$
- The gradients for $W_{xh}$ and $W_{hh}$ are more complex because they affect the hidden states across multiple time steps. We have to accumulate the gradients from all time steps.
    $$
    \frac{\partial C}{\partial W_{xh}} = \sum_{t=1}^{T} \frac{\partial C_t}{\partial W_{xh}}
    $$
    $$
    \frac{\partial C}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial C_t}{\partial W_{hh}}
    $$

This accumulation is what allows the RNN to learn from the entire sequence, capturing dependencies over time.


## Python Implementation

Here's a simple implementation of a basic RNN in Python using NumPy:

```python
import numpy as np

class RNN:
    def __init__(self, hidden_size, vocab_size, learning_rate=0.01):
        '''
        Initialize the RNN layer.
        hidden_size: Number of neurons in the hidden state.
        vocab_size: Number of unique tokens in the input data.
        '''

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lr = learning_rate

        # 1. Initialize weights and biases
        # we use small random values * 0.01 to avoid exploding gradients
        # W_xh: From input x to hidden h, the input x is one-hot encoded
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        # W_hh: From hidden h(t-1) to hidden h(t)
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        # W_hy: From hidden h to output y
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def forward(self, inputs, h_prev):
        '''
        For a full sequence.

        inputs: List of token indices for the sequence.
        h_prev: Previous hidden state (initially zeros).

        Returns:
        y_preds: List of softmax probabilities for each time step.
        h_final: The final hidden state after processing the sequence.
        cache: Values needed for backpropagation.
        '''

        xs, hs, ys, y_preds = {}, {}, {}, {} # these are x: input, h: hidden state, y: output before softmax, y_pred: after softmax
        hs[-1] = np.copy(h_prev) # initial hidden state at t=-1
        for t in range(len(inputs)):
            # 1. One-hot encode input
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1

            # 2. Compute hidden state
            # h_t = tanh(W_xh * x_t + W_hh * h_(t-1) + b_h)
            zt = np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh # dimensions: (hidden_size, 1)
            hs[t] = np.tanh(zt)

            # 3. Compute output
            # y_t = W_hy * h_t + b_y
            ys[t] = np.dot(self.Why, hs[t]) + self.by

            # 4. Softmax to get probabilities
            y_preds[t] = self._softmax(ys[t])

        cache = (xs, hs, y_preds)
        return y_preds, hs[len(inputs)-1], cache # hs[len(inputs)-1] is the final hidden state

    def compute_cost(self, y_preds, targets):
        '''
        Calculate the total cross-entropy for the sequence.
        y_preds: Predicted probabilities for each time step.
        targets: True token indices for the sequence.
        '''

        total_cost = 0
        for t in range(len(targets)):
            prob_of_correct_class = y_preds[t][targets[t], 0]
            total_cost += -np.log(prob_of_correct_class + 1e-9) # add small value to avoid log(0)

        return total_cost / len(targets)  # average cost per time step

    def backpropagation(self, targets, cache):
        '''
        Performs BPTT to compute gradients.
        targets: True token indices for the sequence.
        cache: (xs, hs, y_preds) from forward pass.
        '''
        xs, hs, y_preds = cache
        # Initialize gradients
        self.dWxh, self.dWhh, self.dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.dbh, self.dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        # Gradient of the hidden state
        dh_next = np.zeros((self.hidden_size, 1))

        # --- Loop backward through time steps ---
        for t in reversed(range(len(targets))):
            # 1. Output gradients
            # Gradient of softmax + cross-entropy loss (AL - Y)
            dy = np.copy(y_preds[t])
            dy[targets[t]] -= 1  # derivative of loss w.r.t. y before softmax

            # 2. Gradients for output layer Why and by
            self.dWhy += np.dot(dy, hs[t].T) # vocab x hidden
            self.dby += dy

            # 3. Gradient for hidden state dh_t
            dh = np.dot(self.Why.T, dy) + dh_next  # dh_next is from next time step
            # 4.  Tanh Activation Zht
            # dtanh = 1 - tanh^2(zt) = 1 - h_t^2
            dzt = (1 - hs[t] * hs[t]) * dh  # element-wise multiplication (hidden, 1)
            # 5. Gradients for Recurrent Layer dWhh, dWxh, dbh
            self.dbh += dzt
            self.dWhh += np.dot(dzt, hs[t-1].T)
            self.dWxh += np.dot(dzt, xs[t].T)

            # 6. Pass gradient to next (previous) time step
            dh_next = np.dot(self.Whh.T, dzt)

        # Clip gradients to prevent exploding gradients
        for grad in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
            np.clip(grad, -5, 5, out=grad)
        
    def update_parameters(self):
        '''
        Performs a single gradient descent update.
        '''
        self.Wxh -= self.lr * self.dWxh
        self.Whh -= self.lr * self.dWhh
        self.Why -= self.lr * self.dWhy
        self.bh -= self.lr * self.dbh
        self.by -= self.lr * self.dby

    def sample(self, seed_idx, h_prev, length=20):
        '''
        This function generates a sequence of indices given a seed index and initial hidden state.
        '''
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []

        for t in range(length):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
            z = np.dot(self.Why, h) + self.by
            y = self._softmax(z)

            idx = np.random.choice(range(self.vocab_size), p=y.ravel())

            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            indices.append(idx)
            h_prev = h
        return indices
```


## The Big Problem with BPTT

This process of multiplying gradients through many time steps leads to a famous issue:
- **Vanishing Gradients**: If the gradient is a small number _(e.S., $<1$_), multiplying it many times makes it shrink towards zero. The early layers _(like $t=1$)_ get no gradient signal and thus **stop learning**. The network **forgets** what it's supposed to learn from the beginning of the sequence.
- **Exploding Gradients**: Conversely, if the gradient is a large number _(e.S., $>1$_), multiplying it many times makes it grow exponentially. This can cause the weights to become very large and unstable, leading to **divergent behavior** during training.

The difficulty in learning long-range dependencies _(long-term memory)_ is the main weakness of the simple RNN.