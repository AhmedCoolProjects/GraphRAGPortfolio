# Graph Masked AutoEncoder (GMAE)

**GraphMAE**, a method that brings **generative** learning to GNNs. It is inspired by successful language models like **BERT**.

It works in three steps:

1. **Masking**: Instead of showing the model the whole graph, we randomly hide _(mask)_ a portion of the node features.
2. **Encoding**: The GNN encoder looks at this incomplete graph. It has to understand the context of the remaining nodes and the structure to figure out what is missing.
3. **Reconstruction**: The decoder's job is to predict the exact features of the nodes that were masked.

## GAE vs. GMAE

The main difference lies in **what they reconstruct** and **how they learn**.

| Feature          | GAE                                                | GMAE                                              |
| ---------------- | -------------------------------------------------- | ------------------------------------------------- |
| **Primary Goal** | Reconstruct the Structure $A$                      | Reconstruct the Features $X$                      |
| **Input**        | Complte Graph ($A, X$)                             | Masked Graph ($A, \tilde{X}_{masked}$)            |
| **Decoder**      | Usually simple **dot product** _(link prediction)_ | Usually **GNN** or **MLP** _(Feature Regression)_ |
| **Analogy**      | "Who are friends with whome?"                      | "What is the missing word in the sentence?"       |

While GAEs are great for finding missing links, GMAEs are often more powerful at learning robust node embeddings because the model is forced to understand the **content** of the nodes, not just their connections.

<details>
<summary>Why Masking?</summary>

Hiding information forces the model to move from **memorizing** to **reasoning**.

</details>

## Forward Pass

### Inputs

We start with the same standard graph inputs as before:

- **Node Features ($X$)**: The content ($N \times F$).
- **Adjacency Matrix ($A$)**: The structure ($N \times N$).

### The `Mask`

Before anything enters the neural network, we apply the mask.

1. **Selection**: We randomly select a subset of nodes to be masked, let's call their indices $\mathcal{M}$.
2. **Replacement**: For every node $i \in \mathcal{M}$, we replace its feature vector $x_i$ with a learnable parameter vector _(token)_, let's call it $e_{[MASK]}$.

Mathematically, the new input feature matrix $\tilde{X}$ looks like this:

$$
\tilde{x_i} = \begin{cases}
e_{[MASK]} & \text{if } i \in \mathcal{M} \\
x_i & \text{if } i \notin \mathcal{M}
\end{cases}
$$

:::note

The Adjacency matrix $A$ remains **unchanged**, so the "masked" node can still receive messages.

:::

### The Math

#### Encoder (The "Context Learner")

We feed corrupted $\tilde{X}$ and the original structure $A$ into a GNN _(like GCN or GAT)_.

$$
Z = \text{GNN}_{enc}(\tilde{X}, A)
$$

- **Goal**: The encoder tries to generate an embedding $z_i$ for every node $i$.
- **For masked nodes**: Since their own features are effectively "blank", their embedding $z_i$ is constructed entirely from aggregating information from their neighbors.

#### Decoder (The "Reconstructor")

Now we have latent embeddings $Z$. We need to map these back to the original feature space dimension _($F$)_. Unlike standard GAEs which use a simple dot product, GMAEs often use another GNN layer or a Multi-Layer Perceptron (MLP) as the decoder.

$$
\hat{X} = \text{GNN}_{dec}(Z)
$$

- **Output**: $\hat{X}$ is a matrix of size $N \times F$. It contains the predicted feature vectors for all nodes.

### Loss Function

Here is the critical difference from the link prediction GAE. We don't care about the edges; we care about the data inside the nodes.

We compare the **Predicted Features ($\hat{X}$)** against the **Original Features ($X$)**.

Crucially, we usually calculate the loss **only on the masked nodes** _(similar to how BERT learns)_. We don't need to relearn the features we already saw.

$$
\mathcal{L} = \sum_{i \in \mathcal{M}} \text{Distance}(x_i, \hat{x}_i)
$$

where $\text{Distance}$ is the chosen distance metric _(e.g., MSE, MAE, etc.)_.

In modern GMAEs _(like GraphMAE)_, we often use a specific loss called **Scaled Cosine Error (SCE)** instead of just **MSE**.

**Why?** MSE focuses on the magnitude of vectors, but in high-dimensional spaces, the _direction (angle)_ often matters more for semantics.

**The Equation for a single masked node $\mathcal{i}$:**

$$
\mathcal{L}_{\mathcal{i}} = \left(1 - \frac{\hat{x}_{\mathcal{i}} \cdot x_{\mathcal{i}}}{\|\hat{x}_{\mathcal{i}}\| \|x_{\mathcal{i}}\|}\right)^p
$$

where:

- $\hat{x}_{\mathcal{i}}$ is the predicted feature vector for node $\mathcal{i}$.
- $x_{\mathcal{i}}$ is the original feature vector for node $\mathcal{i}$.
- $\|\hat{x}_{\mathcal{i}}\|$ is the magnitude of the predicted feature vector.
- $\|x_{\mathcal{i}}\|$ is the magnitude of the original feature vector.
- The fraction is the **Cosine Similarity** _(ranges from -1 to 1)_.
- The power $p \ge 1$ acts as a scaling factor to sharpen the focus on hard samples.

## PyTorch Implementation

Let's import the required libraries.

```python title="Required Libraries"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch Geometric Imports
try:
    from torch_geometric.nn import GATConv
except ImportError:
    print("Please install torch-geometric: pip install torch-geometric")
    exit()

# Set seeds
torch.manual_seed(42)
```

Let's create a helper function to mask graph node features and define the GMAE model.

```python title="GraphMAE"
class GMAE(nn.Module):
    def __init__(self, in_dim, hid_dim, latent_dim, heads=4, dropout=0.2):
        super(GMAE, self).__init__()

        self.dropout_rate = dropout

        # The [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        nn.init.xavier_uniform_(self.mask_token)

        # Encoder (GAT)
        self.enc_gat1 = GATConv(in_dim, hid_dim, heads=heads, dropout=dropout)
        self.enc_gat2 = GATConv(hid_dim * heads, latent_dim, heads=1, concat=False, dropout=dropout)

        # Decoder (GAT)
        self.dec_gat1 = GATConv(latent_dim, hid_dim, heads=heads, dropout=dropout)
        self.dec_gat2 = GATConv(hid_dim * heads, in_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # 1. Encoder
        # x is already masked

        z = self.enc_gat1(x, edge_index)
        z = F.elu(z)
        z = F.dropout(z, p=self.dropout_rate, training=self.training)

        z = self.enc_gat2(z, edge_index)
        # z is the latent space

        # 2. Decoder
        recon = self.dec_gat1(z, edge_index)
        recon = F.elu(recon)
        recon = F.dropout(recon, p=self.dropout_rate, training=self.training)

        recon = self.dec_gat2(recon, edge_index)
        # recon is the reconstructed features

        return recon

    def mask_features(self, x, mask_rate=0.3):
        '''
        Randomly replaces node features with the [MASK] token.
        Returns:
        - masked_x: The feature matrix with some rows replaced by the mask token.
        - mask_mask: Boolean tensor indicating which nodess were masked.
        '''
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes)
        num_mask = int(mask_rate * num_nodes)

        # Indices to mask
        mask_idx = perm[:num_mask]

        masked_x = x.clone()

        masked_x[mask_idx] = self.mask_token.expand(num_mask, -1)

        return masked_x, mask_idx
```

Let's train our model.

```python title="Training Loop"
# we suppose that we have our data from pytorch geometric planetoid
def train_gmae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    HID_DIM = 64
    LATENT_DIM = 32
    HEADS = 4
    LR = 0.001
    EPOCHS = 200
    MASK_RATE = 0.3

    model = GMAE(in_dim=num_features,
                 hid_dim=HID_DIM,
                 latent_dim=LATENT_DIM,
                 heads=HEADS).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4) # weight_decay is equivalent to L2 regularization

    criterion = nn.MSELoss()

    print(f'Starting GMAE Training on {device}...')
    loss_history = []

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        masked_x, mask_idx = model.mask_features(data.x, mask_rate=MASK_RATE)

        recon_x = model(masked_x, data.edge_index)

        loss = criterion(recon_x[mask_idx], data.x[mask_idx])

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1:03d} | Loss: {loss.item():.6f}')

    print("\nTraining complete.")

```

:::tip GATConv

- It does detect if the `edge_index` contains **self-loops** or not, if not it has a default parameter `add_self_loops=True`.
- The `mask_token` is initialized with a random vector and is learnable. It allows the NN to mathematically optimize the best possible representation for "missing data".

:::

## Reference

Find the original paper [here](https://dl.acm.org/doi/pdf/10.1145/3534678.3539321)
