# AutoEncoders

An Autoencoder is a type of artificial neural network used to learn **efficient codings of data** in an unsupervised manner. The aim of an autoencoder is to learn a representation _(encoding)_ of a set of data, typically for **dimensionality reduction**, by training the network to ignore signal "noise".

Essentially, it tries to copy its input to its output. However, the network is forced to compress the input into a lower-dimensional space, which is then decoded back to the original space.

## Architecture

An autoencoder consists of three main parts that work together like a compression and decompression system:

### 1. Encoder

This part takes the input data _(like an image or a row of numbers)_ and compresses it. It passes the data through neural network layers that progressively get smaller, forcing the information into a more compact form.

### 2. Bottleneck _(Latent Space)_

This is the central layer with the fewest neurons. It holds the **encoded representation** of the input. Because it is small, it acts as a filter, allowing only the most essential features to pass through while blocking the less important details.

:::info Note

If the latent space is too big _(same size as the input or bigger)_ the autoencoder will not be able to learn a good representation of the input, instead it will just copy the input to the output, with all the noise and details.

:::

### 3. Decoder

This part takes the compressed code from the latent space and tries to reconstruct the original input. It uses layers that progressively get larger until they match the size of the initial input. The goal is to recreate the input as accurately as possible.

---
An Autoencoder is just **two** Neural Networks _([check the NN toturial](../neural-networks/))_ glued together back-to-back.
- **Network 1 (Encoder)**: Shrinks the input.
- **Network 2 (Decoder)**: Expands the bottleneck.

## Training Process

The goal is to make the output $\hat{X}$ look exactly like the input $X$. To do so, we need a loss function that measures the difference between the two. The most common loss function is the **Mean Squared Error** _(MSE)_. In the context of autoencoders, this is called **Reconstruction Loss**.

$$
L(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2
$$

### The "Push and Pull" of Training

Training an autoencoder is a balancing act:
1. **The Constraint**: The **bottleneck** forces the network to throw away information.
2. **The Goal**: The **reconstruction loss** forces the network to keep information.

Because the network wants to minimize that **MSE** to zero, but it can't just copy the data _(due to the bottleneck)_, it is forced to cheat. It cheats by finding the **correlations** and **patterns** in the data.

<details>
<summary>Example</summary>

In the case of images, if every time pixel A and pixel B are black, the bottleneck doesn't need to store both of them. It just stores "Pattern A+B" and the Decoder knows how to unpack that.

</details>

### Backpropagation

Same as any other Neural Network, the autoencoder is trained using **Gradient Descent**.

$$
W_{new} = W_{old} - \alpha \frac{\partial L}{\partial W}
$$

where $\alpha$ is the **learning rate**.

Thus the full training loop:
1. **Forward Pass**: Input $X$ $\rightarrow$ Encoder $\rightarrow$ Bottleneck $\rightarrow$ Decoder $\rightarrow$ Output $\hat{X}$
2. **Calculate Loss**: $L(X, \hat{X})$ using the **MSE** loss function.
3. **Backpropagation**: Calculate the gradients for Encoder and Decoder.
4. **Update**: Adjust the weights to reduce the loss.
5. **Repeat**: Do this thousands of times until the reconstruction is perfect _(or close enough)_.