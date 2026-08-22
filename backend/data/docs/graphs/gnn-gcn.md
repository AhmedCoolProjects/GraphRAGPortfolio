# Graph Convolutional Networks (GCN)

Before we elaborate on why **GNNs** are important, let's first understand the problem with Standard Networks.

Most classic NNs _(like CNN or RNN)_ assume your data has a strict regular structure:

- **Images _(Euclidean Space)_**: Pixel values are arranged perfectly in a grid. _("up", "down", "left", "right" are valid directions)_
- **Text/Audio _(Sequence)_**: Data comes in a sequence. _(one word after another)_

Many real-world problems don't fit into grids or lines _(Social Networks, Molecules, Servers Connections, etc.)_

We needed an architecture that could handle:

1. **Non-Euclidean** data.
2. **Size Invariance** where it handles graphs of any size.
3. **Permutation Invariance** since graphs are isomorphic _(same graph but different node ordering)_.

The core solution **GNNs** propose is **Message Passing**. Instead of scanning the whole graph, nodes learn by "talking" to their neighbors.

## Forward Pass

There are 2 main steps in the forward pass:

### 1. Message Passing

Message Passing _(Aggregation)_ where every node looks _only_ at its immediate neighbors and "gathers" their information.

This aggregation of neighbors features can be done in many ways _(sum, max, mean, etc.)_.

### 2. Update

The node combines the gathered information with its own current information to update itself.

This update step can also be done in many ways _(sum, max, mean, etc.)_. But these simple operations will lead to an explosion of numbers _(high variance)_. We need a NN in the middle.

We sum the node own features and the aggregated neighbors features. We pass this to a NN to update the node features by multiplication with a weight matrix and an activation function on the result.

$$
\text{New Node A} = ReLU(W \times (\text{Node A} + \text{Aggregated Neighbors}))
$$

---

In one **GNN Layer**, these two steps are applied to every node in the graph.

:::tip
In GNNs, applying 1 layer on the graph is equivalent to a node seeing its immediate neighbors. Applying 2 layers leads to a node seeing its neighbors' neighbors, and so on.

- **1 GNN Layer** = **1 Hop**
- **2 GNN Layers** = **2 Hops**
- **3 GNN Layers** = **3 Hops**
- **...**

:::

### Math

To represent a graph, we need **two main matrices**:

- **Feature Matrix _($X$)_**: Each row is a node's features.
  - Sahpe: _($N \times F$)_ where _$N$_ is the number of nodes and _$F$_ is the number of features per node.
- **Adjacency Matrix _($A$)_**: Showing who's connected to whom.
  - Shape: _($N \times N$)_ where _$N$_ is the number of nodes.
  - If Node _$i$_ is connected to Node _$j$_, then _$A_{ij} = 1$_ otherwise _$A*{ij} = 0$*.

**The Magic of Matrix Multiplication**

The reason why GNNs are fast is that we don't use a for loop to sum neighbors. We just multiply matrices.

Let's say we have the matrix _($A$)_:

$$
A = \begin{bmatrix}
    0 & 1 & 1 \\
    1 & 0 & 0 \\
    1 & 0 & 0
\end{bmatrix}
$$

and the matrix _($X$)_:

$$
X = \begin{bmatrix}
    x_A \\
    x_B \\
    x_C
\end{bmatrix}
$$

#### 1. Message Passing

We simply multiply the adjacency matrix _($A$)_ with the feature matrix _($X$)_ to get the aggregated neighbors features for each node.

$$
A \times X = \begin{bmatrix}
    x_B + x_C \\
    x_A \\
    x_A
\end{bmatrix}
$$

#### 2. Self-Awareness

To include the node's own features, we can add the _Identity_ matrix _($I$)_ to the adjacency matrix _($A$)_ before the multiplication.

$$
A + I = \begin{bmatrix}
    1 & 1 & 1 \\
    1 & 1 & 0 \\
    1 & 0 & 1
\end{bmatrix}
$$

And then:

$$
\tilde{A} \times X = \begin{bmatrix}
    x_A + x_B + x_C \\
    x_A + x_B \\
    x_A + x_C
\end{bmatrix}
$$

#### 3. Normalization

In **GCNs**, we normalize the adjacency matrix _($A$)_ by the degree of each node.

We normalize by $\tilde{D^{-1/2}}$ where $\tilde{D}_{ii} = \text{deg}(i)$, because:

- We want the operator to be symmetric _(important for spectral convolution)_.
- We want to treat node $i$ and node $j$ fairly _(avoid degree bias)_.
- We want the operator to behave like the graph $Laplacian$.
- We want stable gradients and smooth feature propagation.

The formula:

$$
\boxed{\tilde{A} = {\tilde{D}^{-1/2}} \cdot (A + I) \cdot {\tilde{D}^{-1/2}}}
$$

is the **only normalization** that satisfies the above properties.

#### 4. Learning

Just summing things isn't "learning". We need a learnable Weight Matrix _($W$)_ to transform these features, just like in a standard Linear Layer with a shape of _($F \times O$)_ where _$F$_ is the number of features and _$O$_ is the number of output features.

$$
\boxed{X_{new} = \sigma(\tilde{A} \cdot X \cdot W)}
$$

<details>
<summary>Example</summary>

Suppose we have:

- $N = 100$ number of nodes in our graph
- Each node currently has $F_{in} = 64$ features
- We want to transform them to $F_{out} = 32$ features

Then after the first layer, our $X_{new}$ will have a shape of _($100 \times 32$)_.

</details>

### Implementation

We will create a class called `GCNLayer` and implement the forward pass of it.

```python title="GCN Layer"
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, use_relu=True):
        super().__init__()
        # Define the learnable weights here
        self.linear = nn.Linear(in_feats, out_feats) # this will create W and b

    def forward(self, node_feats, adj_matrix):
        # 1. transform
        trans_feats = self.linear(node_feats)
        # 2. normalization
        I = torch.eye(adj_matrix.shape[0])
        A_tilde = adj_matrix + I
        D_tilde_sqrt = torch.diag(torch.pow(A_tilde.sum(dim=1), -0.5))
        A_norm = D_tilde_sqrt @ A_tilde @ D_tilde_sqrt
        # 3. message passing
        message = A_norm @ trans_feats
        # 4. update
        if self.use_relu:
            return F.relu(message)
        return message

```

## Backpropagation

We assume we receive the gradient of the loss $L$ w.r.t. the output of the GCN layer $Z$. We call this incoming gradient matrix $G$:

$$
G = \frac{\partial L}{\partial Z}
$$

( Shape: $$N \times F_{out}$$ )

We need to calculate two gradients to continue backpropagation:

- $\frac{\partial L}{\partial W}$ _(to update our weights)_
- $\frac{\partial L}{\partial X}$ _(to pass the error to the previous layer)_

We know $ Z = \tilde{A} \cdot X \cdot W $, so:

$$
\frac{\partial L}{\partial W} = (\tilde{A} \cdot X)^T \cdot G
$$

$$
\frac{\partial L}{\partial X} = \tilde{A}^T \cdot G \cdot W^T
$$

and since $\tilde{A}$ is symmetric, we can write:

$$
\frac{\partial L}{\partial X} = \tilde{A} \cdot G \cdot W^T
$$

:::info Intuition

1. **Forward Pass**: Nodes send **features** to their neighbors.
2. **Backward Pass**: Nodes send **errors** back to their neighbors.

:::

---

## PyTorch Implementation

```python title="GCN"
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Layer 1
        self.gcn1 = GCNLayer(input_dim, hidden_dim, use_relu=True)
        # Layer 2
        self.gcn2 = GCNLayer(hidden_dim, output_dim, use_relu=False)

    def forward(self, node_feats, adj_matrix):
        x = self.gcn1(node_feats, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        return x

```

Now we need to create a training loop where we will:

1. Initialize the model
2. Define the Optimizer
3. Define the Loss Function
4. Define the Training Loop

```python title="Training Loop"
import torch.optim as optim

# --- Dummy Data ---
# 4 Nodes, 3 Features each
node_features = torch.tensor([
    [1.0, 0.0, 0.0], # Node 0
    [0.0, 1.0, 0.0], # Node 1
    [0.0, 0.0, 1.0], # Node 2
    [1.0, 1.0, 0.0]  # Node 3
])

# Adjacency Matrix (4x4)
adj_matrix = torch.tensor([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0]
], dtype=torch.float32)

# Labels: We want to classify the nodes into Class 0 or Class 1
# Let's say Node 0 & 1 are Class 0, Node 2 & 3 are Class 1
labels = torch.tensor([0, 0, 1, 1])

# --- Initialize Model ---
# Input: 3 features -> Hidden: 4 features -> Output: 2 classes
model = GCN(input_dim=3, hidden_dim=4, output_dim=2)

# --- Optimizer & Loss ---
# Optimizer handles the parameter updates (Gradient Descent)
optimizer = optim.Adam(model.parameters(), lr=0.01)
# CrossEntropyLoss is standard for classification
criterion = nn.CrossEntropyLoss()

print("Starting Training...\n")

for epoch in range(100): # Run for 100 loops
    model.train() # Set model to training mode

    # 1. Zero Gradients
    # Clear old gradients from the previous step
    optimizer.zero_grad()

    # 2. Forward Pass
    # Get predictions from the model
    output = model(node_features, adj_matrix)

    # 3. Calculate Loss
    # Compare output with actual labels
    loss = criterion(output, labels)

    # 4. Backward Pass (The Math Magic)
    # PyTorch calculates gradients (dLoss/dW) for all weights automatically
    loss.backward()

    # 5. Update Weights
    # Adjust weights: W_new = W_old - (lr * gradient)
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# Final Prediction
print("\nFinal Node Classifications:")
final_output = model(node_features, adj_matrix)
predicted_classes = final_output.argmax(dim=1)
print(f"Predicted: {predicted_classes}")
print(f"Actual:    {labels}")

```

**Output**

```
Starting Training...

Epoch 0 | Loss: 0.6986
Epoch 10 | Loss: 0.6926
Epoch 20 | Loss: 0.6892
Epoch 30 | Loss: 0.6846
Epoch 40 | Loss: 0.6763
Epoch 50 | Loss: 0.6620
Epoch 60 | Loss: 0.6399
Epoch 70 | Loss: 0.6097
Epoch 80 | Loss: 0.5725
Epoch 90 | Loss: 0.5304

Final Node Classifications:
Predicted: tensor([0, 0, 1, 1])
Actual:    tensor([0, 0, 1, 1])
```

## Notebook

Check [this](https://gist.github.com/AhmedCoolProjects/6344dd2d104fc068d05cf50b5b08f828) notebook for a more detailed explanation.
