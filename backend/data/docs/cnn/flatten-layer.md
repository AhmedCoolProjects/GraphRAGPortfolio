# Flatten Layer

After several convolutional and pooling layers, we end up with a set of feature maps that capture the important patterns in the input image. However, these feature maps are still in a 3D format (height, width, number of filters). And basically when we have a batch of images, we end up with a 4D format (batch size, height, width, number of filters).

To feed this data into a traditional Dense NN for classification, we need to convert this 4D volume into a 2D format (batch size, features). This is where the **Flatten Layer** comes in.

The Flatten Layer takes each 3D feature map for an image and flattens it into a 1D vector. For example, if after the last pooling layer we have feature maps of size $7\times7\times64$ (height, width, number of filters), the Flatten Layer will convert this into a vector of size $7 \times 7 \times 64 = 3136$.

The flattening process is straightforward: We start by flattening the feature maps (number of filters), then the width, and finally the height. This way, the spatial relationships are preserved in the flattened vector.

## Python Implementation

```python title="flatten_layer.py"
class Flatten_Layer:
    '''
    A layer to flatten the 4D output of the Pool Layer into a 2D vector to be fed into the Dense NN.
    '''
    def __init__(self):
        self.cache = None

    def forward_pass(self, A_prev: np.ndarray) -> np.ndarray:
        '''
        Converts (m, H, W, C) to (H*W*C, m)

        Returns:
            A: np.ndarray -- Flattened output of shape (H*W*C, m)
            cache: Original shape for backpropagation
        '''
        self.cache = A_prev.shape
        (m, H, W, C) = A_prev.shape

        A = A_prev.reshape(m, -1).T  # (H*W*C, m), -1 infers the size

        return A, self.cache

    def backpropagation(self, dA: np.ndarray) -> np.ndarray:
        '''
        Reshapes dA back to the original shape stored in cache.

        Returns:
            dA_prev: np.ndarray -- Gradient reshaped to original dimensions
        '''
        original_shape = self.cache
        dA_prev = dA.T.reshape(original_shape)  # (m, H, W, C)

        return dA_prev
```
