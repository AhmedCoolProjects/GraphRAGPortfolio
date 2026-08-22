# Pooling Layer

This lyaer has one simple job: **shrink the feature maps** coming from the convolutional layer.

After the convolutional layer, we have a big 3D volume _(height, width, number of filters)_. This is a lot of data. We want to make it smaller and more manageable, both to save computation and to make the network more robust to small changes in the image _(like shifting the object a few pixels)_.

The pooling layer does this by sliding a small window _(usually $2\times2$)_ over each feature map and taking a summary of that window.

### How Pooling Works

The most common type of pooling is **Max Pooling**. It's a simple, fixed operation:
1. **No Weights, No Biases**: This layer does not learn anything. It has 0 parameters.
2. **Sliding Window**: It slides a $2\times2$ window over the input feature map, usually with a **stride of 2** (meaning it moves 2 pixels at a time, so the windows do not overlap).
3. **The Rule**: From the $4$ values in the $2\times2$ window, it takes the **maximum** value.

The resulted **Pooled Map** is half the height and half the width of the original feature map, but it keeps the most important information (the strongest activations).

:::info Note
- **CONV Layer**: _(with ReLU)_ finds patterns in the image.
- **POOL Layer**: shrinks and summarizes the feature maps.
---
The resulting feature maps from the conv layer are not always evenly divisible by 2, thus the pooling layer with a $2\times2$ window and stride of 2 won't fit perfectly. In such cases, we can use two ways of padding:
- **VALID Padding**: Simply ignore the extra row/column that doesn't fit.
- **SAME Padding**: Add extra row/column of zeros around the feature map so that it fits perfectly so we don't lose any data.
:::

### Backpropagation

Backpropagation in pooling layers is straightforward since there are no weights to update. During the backward pass, the gradient is passed **only to the position of the maximum value** that was selected during the forward pass. All other positions receive a gradient of zero.

Let's say we got $(m, H, W, C)$ from the previous conv layer to the current pool layer, where: $m$ is the batch size, $H$ is the height of the feature maps, $W$ is the width of the feature maps, and $C$ is the number of channels _(filters)_.

Let $A_{prev}$ be the input to the pooling layer with shape $(m, H, W, C)$ and $A$ be the output of the pooling layer with shape $(m, H_{out}, W_{out}, C)$.

Let $dA$ be the gradient of the loss with respect to the output $A$, and we want to compute $dA_{prev}$.

To do this, we:
1. Initialize $dA_{prev}$ with zeros, having the same shape as $A_{prev}$.
2. For each position in the output $A$, we find the corresponding $2\times2$ window in $A_{prev}$.
3. Identify the position of the maximum value in that window based on the cache from the forward pass.
4. Assign the gradient from $dA$ to that position in $dA_{prev}$.
5. All other positions in the window receive a gradient of zero.
6. Repeat this for all examples in the batch and all channels.

Let's take an example of one feature map of size $4\times4$ being pooled to a $2\times2$ feature map using max pooling with a $2\times2$ window and stride of 2.

$$
A_{prev} =
\begin{bmatrix}
1 & 3 & 2 & 4 \\
5 & 6 & 1 & 2 \\
3 & 2 & 8 & 1 \\
0 & 1 & 4 & 5
\end{bmatrix}
\quad
\Rightarrow
\quad
A =
\begin{bmatrix}
6 & 4 \\
3 & 8
\end{bmatrix}
$$

To compute $dA_{prev}$ given $dA$:

$$
dA =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

We initialize $dA_{prev}$ as:

$$
dA_{prev} =
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

We then map the gradients from $dA$ back to $dA_{prev}$ based on the positions of the maximum values in each pooling window:

1. For the top-left window of $A_{prev}$ (covering elements 1, 3, 5, 6), the max is 6 at position (1,1). We assign $dA[0,0] = 1$ to $dA_{prev}[1,1]$.
2. For the top-right window (covering elements 2, 4, 1, 2), the max is 4 at position (0,3). We assign $dA[0,1] = 2$ to $dA_{prev}[0,3]$.
3. For the bottom-left window (covering elements 3, 2, 0, 1), the max is 3 at position (2,0). We assign $dA[1,0] = 3$ to $dA_{prev}[2,0]$.
4. For the bottom-right window (covering elements 8, 1, 4, 5), the max is 8 at position (2,2). We assign $dA[1,1] = 4$ to $dA_{prev}[2,2]$.

After mapping all gradients, we get:

$$
dA_{prev} =
\begin{bmatrix}
0 & 0  & 0 & 2 \\
0 & 1 & 0 & 0 \\
3 & 0 & 4 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$


## Python Implementation


```python title="pool_layer.py"
class Pool_Layer:
    def __init__(self, filter_size=2, stride=2, mode='max'):
        '''
        mode: 'max' for Max Pooling, 'avg' for Mean Pooling
        '''
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

    def _create_mask_from_window(self, x_slice):
        '''
        Helper function for max pooling backpropagation.
        Creates a mask of same shape as x_slice, with a 1 at the position of the max value.
        '''

        mask = (x_slice == np.max(x_slice))
        return mask

    def forward_pass(self, A_prev):
        '''
        A_prev: Input data from the previous layer
        Shape: (m, H_prev, W_prev, C_prev)

        Returns:
        A: Output of the pooling layer
        cache: A tuple of values needed for backpropagation
        '''

        (m, H_prev, W_prev, C_prev) = A_prev.shape
        F = self.filter_size
        S = self.stride

        # Calculate output dimensions
        H_out = int((H_prev - F) / S) + 1
        W_out = int((W_prev - F) / S) + 1

        # Initialize output volume A
        A = np.zeros((m, H_out, W_out, C_prev))

        # --- The Pooling Loop ---
        for i in range(m):  # loop over batch size
            for h in range(H_out):
                for w in range(W_out):
                    for c in range(C_prev):
                        h_start = h * S
                        h_end = h_start + F
                        w_start = w * S
                        w_end = w_start + F

                        # get a slice of the input
                        a_slice = A_prev[i, h_start:h_end, w_start:w_end, c]

                        # Perform pooling operation
                        if self.mode == 'max':
                            A[i, h, w, c] = np.max(a_slice)
                        elif self.mode == 'avg':
                            A[i, h, w, c] = np.mean(a_slice)
        # Store values for backpropagation
        cache = (A_prev, self.filter_size, self.stride, self.mode)

        return A, cache

    def backpropagation(self, dA, cache):
        '''
        dA: Gradient w.r.t the output of the pooling layer, shape: (m, H_out, W_out, C_out)
        cache: Tuple of values from forward pass

        Returns:
        dA_prev: Gradient w.r.t the input of the pooling layer, shape: (m, H_prev, W_prev, C_prev)
        '''

        # 1. Unpack cache
        (A_prev, F, S, mode) = cache

        # 2. Get dimensions
        (m, H_prev, W_prev, C_prev) = A_prev.shape
        (m, H_out, W_out, C_out) = dA.shape

        # 3. Initialize dA_prev
        dA_prev = np.zeros_like(A_prev)  # (m, H_prev, W_prev, C_prev)
        # 4. --- The Backpropagation Loop ---
        for i in range(m):
            x_prev_i = A_prev[i]
            for h in range(H_out):
                for w in range(W_out):
                    for c in range(C_prev):
                        h_start = h * S
                        h_end = h_start + F
                        w_start = w * S
                        w_end = w_start + F

                        da = dA[i, h, w, c]

                        if mode == 'max':
                            x_slice = x_prev_i[h_start:h_end, w_start:w_end, c]
                            # Create mask
                            mask = self._create_mask_from_window(x_slice)
                            # Route the gradient to the position of the max
                            dA_prev[i, h_start:h_end, w_start:w_end, c] += mask * da
                        elif mode == 'avg':
                            # Distribute the gradient evenly
                            shape = (F, F)
                            average = da / (F * F)
                            dA_prev[i, h_start:h_end, w_start:w_end, c] += np.ones(shape) * average
        
        return dA_prev

```
