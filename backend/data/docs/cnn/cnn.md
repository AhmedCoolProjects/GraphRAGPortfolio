# Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a class of deep learning models particularly well-suited for processing data with a **grid-like topology**, such as images. They are designed to automatically and adaptively learn spatial hierarchies of features through the use of convolutional layers, pooling layers, and fully connected layers.

## Why not NNs for Images?

Traditional Neural Networks (NNs) (called **Dense** or **Fully Connected** networks) treat input data as a flat vector, which can lead to several issues when dealing with images:
1. **High Dimensionality**: Images often have a large number of pixels, leading to a high-dimensional input space. This can result in a massive number of parameters in the NN, making it computationally expensive and prone to overfitting. Previously we worked with MNIST images of size $28\times28$ pixels ($784$ features), that's already $784\times64 = 50,176$ parameters just for a first hidden layer with $64$ neurons! For an image of $1000\times1000$ pixels ($1,000,000$ features), this number skyrockets to $1,000,000\times64 = 64,000,000$ parameters just for the first hidden layer!
2. **Loss of Spatial Structure**: By flattening the image, our NN has no idea that two pixels are next to each other. It loses all information about edges, corners, and shapes.

## How CNNs Solve These Issues

CCNs solve these by using new type of layers _(not Dense layers like in traditional NNs)_ at the beginning of the network. Instead of connecting every input pixel to every neuron in the next layer, they use:
- **Convolutional Layers**: These use small **filters** or **kernels** _(like $3\times3$ or $5\times5$ windows)_ that slide across the image to find **local patterns** such as edges, textures, corners, or a patch of color. This is the core new concept in CNNs.
- **Pooling Layers**: These **shrink** the image _(downsample)_ to make it more manageable and help the network recognize a pattern no matter where it appears in the image.

After these new layers find the patterns, the final **flattened** output is fed into a regular **Dense** network just like the one we built for NNs on MNIST.
