# Graph Autoencoders (GAE)

**GAE** is a method that brings **generative** learning to GNNs. "Autoencoders" concept _(compression and reconstruction)_ and "Graph" processor _(GCN or GAT)_.

In standard Autoencoder, we compress input data $X$ into a latent space $Z$ and try to reconstruct $X$ from $Z$.

In a **Graph Autoencoder**, the goal shifts slightly. We use an **encoder** _(like GCNs)_ to compress the graph structure ($A$) and node features ($X$) into latent space embeddings ($Z$).

The **decoder** then uses those embeddings to try and reconstruct the **adjacency matrix ($A$)** _(essentially predicting whether an edge exists between two nodes)_.

## Forward Pass

### Inputs

To process a graph, the GAE layer need two key pieces of information:

- **Node Features ($X$)**: A matrix of size $N \times F$, where $N$ is the number of nodes and $F$ is the number of features per node.
- **Adjacency Matrix ($A$)**: A matrix of size $N \times N$ representing the connections between nodes. Usually, $A_{ij} = 1$ if there is an edge between node $i$ and $j$, and $0$ otherwise.

### The Math

The most common encoder for a GAE is a **GCN**. A simple GAE encoder often has two layers.

1. **Hidden Layer**: $H = ReLU(\tilde{A}XW_{0})$
2. **Embedding Layer**: $Z = \tilde{A}HW_{1}$

where:

- $\tilde{A}$ is the **symmetrically normalized adjacency matrix**.
- $W_{0}$ and $W_{1}$ are the learnable weights for the encoder.
- $H$ is the intermediate hidden representation.
- $Z$ is the final latent embedding matrix ($N \times D$ where $D$ is the embedding dimension).

For the **Encoder (Reconstruction)**, the simplest and most standard decoder is an **Inner Product**. It essentially asks, "Are the embeddings of nodes $i$ and $j$ similar?"

$$
\hat{A} = \sigma(ZZ^{T})
$$

where:

- $ZZ^{T}$ computes the dot product between every pair of node embeddings.
- $\sigma$ is the sigmoid function, squashing the output to the range $[0, 1]$ _(probability of an edge existing)_.
- $\hat{A}$ is the predicted adjacency matrix.

### Outputs

We get two outputs from the GAE:

- **Latent Embeddings ($Z$)**: A low-dimensional representation ($N \times D$). These vectors capture both the node's original features and its structural information in the graph.
- **Reconstructed Adjacency Matrix ($\hat{A}$)**: An $N \times N$ matrix of probabilities. $\hat{A}_{ij}$ represents the probability of an edge between node $i$ and node $j$.

### Loss Function

We do have two matrices $A$ and $\hat{A}$, and we want to measure the difference between them. The most common loss function for GAE is the **Binary Cross-Entropy (BCE)**.

The loss function for a single pair of node ($i, j$) is given by:

$$
L_{ij} = -\left(A_{ij}\log(\hat{A}_{ij}) + (1-A_{ij})\log(1-\hat{A}_{ij})\right)
$$

where:

- $A_{ij}$ is the true adjacency matrix.
- $\hat{A}_{ij}$ is the predicted adjacency matrix.
- $L_{ij}$ is the loss for the pair of nodes $(i, j)$.

The total loss is then given by the sum of the losses for all pairs of nodes:

$$
L = \sum_{i=1}^{N}\sum_{j=1}^{N}L_{ij}
$$

where:

- $L$ is the total loss.
- $N$ is the number of nodes in the graph.
- $L_{ij}$ is the loss for the pair of nodes $(i, j)$.

<details>
<summary>Why not dividing by 2?</summary>

For an undirected graph, the adjacency matrix is symmetric, meaning $A_{ij} = A_{ji}$. This means that we are counting each edge twice in the loss function.

However, in deep learning optimization, we often skip the division by 2.

Imagine we have a function $f(x) = x^2$. The minimum of this function is at $x = 0$. If we have $f(x) = \frac{1}{2}x^2$, the minimum is still at $x = 0$. The only difference is the slope of the function at $x = 0$. This means that the gradient of $f(x)$ at $x = 0$ is the same as the gradient of $f(x)$ at $x = 0$.

</details>

## PyTorch Implementation

```python title="GCN Layer"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set seeds
torch.manual_seed(42)

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, adj):
        support = torch.mm(x, self.W)
        output = torch.mm(adj, support)
        return output

def normalize_adj(adj):
    self_adj = adj + torch.eye(adj.shape[0])
    degrees = torch.sum(self_adj, dim=1)
    d_inv_sqrt = torch.pow(degrees, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(d_mat_inv_sqrt, torch.mm(self_adj, d_mat_inv_sqrt))

```

```python title="GAE Encoder"
class GAE_Encoder(nn.Module):
    '''
    Standard GCN Encoder.
    Input: (NxF), (NxN)
    Output: (NxD)
    '''
    def __init__(self, in_dim, hid_dim, latent_dim, dropout=0.0):
        super(GAE_Encoder, self).__init__()
        self.gc1 = GCNLayer(in_dim, hid_dim)
        self.gc2 = GCNLayer(hid_dim, latent_dim)
        self.dropout_rate = dropout

    def forward(self, x, adj):
        # layer 1
        hidden = F.relu(self.gc1(x, adj))
        hidden = F.dropout(hidden, self.dropout_rate, training=self.training)

        # layer 2
        z = self.gc2(hidden, adj)
        return z
```

```python title="GAE Decoder"
class InnerProductDecoder(nn.Module):
    '''
    Standard Inner Product Decoder.
    Input: (NxD)
    Output: (NxN)
    '''
    def __init__(self, dropout=0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout_rate = dropout

    def forward(self, z):
        adj_logits = torch.mm(z, z.t())
        return adj_logits

```

```python title="GAE Model"
class GAE(nn.Module):
    def __init__(self, in_dim, hid_dim, latent_dim, dropout=0.0):
        super(GAE, self).__init__()
        self.encoder = GAE_Encoder(in_dim, hid_dim, latent_dim, dropout)
        self.decoder = InnerProductDecoder(dropout)

    def forward(self, x, adj):
        # encoder
        z = self.encoder(x, adj)

        # decoder
        adj_logits = self.decoder(z)

        return adj_logits, z
```

```python title="Training Loop"
def train_gae():
    # we suppose we have x and adj
    in_dim = x.shape[1]
    hid_dim = 32
    latent_dim = 16

    adj_norm = normalize_adj(adj)

    model = GAE(in_dim, hid_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.BCEWithLogitsLoss()

    print('Starting GAE Training...')
    loss_history = []

    model.train()


    for epoch in range(200):
        optimizer.zero_grad()

        # Forward
        adj_logits, z = model(x, adj_norm)

        # Loss
        loss = criterion(adj_logits.view(-1), adj.view(-1)) # -1 to flatten the matrix

        # Backpropagation
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
```
