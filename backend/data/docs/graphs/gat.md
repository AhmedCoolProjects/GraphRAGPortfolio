# Graph Attention Networks (GAT)

Graph Attention Networks (GATs) were introduced by Veličković et al. in 2018 as a novel neural network architecture that operates on graph-structured data, leveraging masked self-attention layers to address the shortcomings of prior methods based on graph convolutions or their approximations.

## Why GATs?

Before GATs, Graph Convolutional Networks (GCNs) were the standard. GCNs combine feature information from neighbors using a fixed normalization (usually based on node degrees). While effective, GCNs have limitations:

1.  **Isotropic Aggregation**: GCNs treat all neighbors as equally important (or weighted only by structural properties like degree), regardless of the actual feature content.
2.  **Transductive Nature**: Many early spectral-based GCNs were difficult to apply to unseen nodes (inductive setting) because the Laplacian eigenbasis depends on the specific graph structure.

GATs introduce an **attention mechanism** that allows the model to learn _importance weights_ for each neighbor. This means the model can focus on the most relevant neighbors for a given task, making the aggregation **anisotropic**.

## GAT vs GCN

| Feature         | GCN (Graph Convolutional Network)         | GAT (Graph Attention Network)                     |
| :-------------- | :---------------------------------------- | :------------------------------------------------ |
| **Aggregation** | Isotropic (Fixed weights based on degree) | Anisotropic (Learnable weights based on features) |
| **Weights**     | $c_{ij} = \frac{1}{\sqrt{d_i d_j}}$       | $\alpha_{ij}$ learned via attention               |
| **Computation** | Cheaper (Matrix multiplication)           | More expensive (Pairwise attention scores)        |
| **Flexibility** | Less flexible to edge nuances             | Can ignore irrelevant edges                       |

## The Math: Forward Pass

Let's break down the operations in a single GAT layer.

**Input**: A set of node features $\mathbf{h} = \{ \vec{h}_1, \vec{h}_2, \dots, \vec{h}_N \}$, where $\vec{h}_i \in \mathbb{R}^F$.
**Output**: A new set of node features $\mathbf{h}' = \{ \vec{h}'_1, \vec{h}'_2, \dots, \vec{h}'_N \}$, where $\vec{h}'_i \in \mathbb{R}^{F'}$.

### 1. Linear Transformation

First, a shared linear transformation (weight matrix $\mathbf{W} \in \mathbb{R}^{F' \times F}$) is applied to every node.
$$ \vec{z}\_i = \mathbf{W} \vec{h}\_i $$

### 2. Self-Attention Mechanism

We compute a pair-wise _attention coefficient_ $e_{ij}$ that indicates the importance of node $j$'s features to node $i$.
$$ e\_{ij} = a(\mathbf{W}\vec{h}\_i, \mathbf{W}\vec{h}\_j) $$

In the standard GAT implementation, the attention mechanism $a$ is a single-layer feedforward neural network, parametrized by a weight vector $\vec{\mathbf{a}} \in \mathbb{R}^{2F'}$, followed by a LeakyReLU nonlinearity.
$$ e\_{ij} = \text{LeakyReLU}\left( \vec{\mathbf{a}}^T [\mathbf{W}\vec{h}_i \, \| \, \mathbf{W}\vec{h}_j] \right) $$
where $\|$ represents concatenation.

### 3. Normalization (Softmax)

To make coefficients comparable across different nodes, we normalize them using the softmax function over the neighbors $\mathcal{N}_i$ of node $i$.
$$ \alpha*{ij} = \text{softmax}\_j(e*{ij}) = \frac{\exp(e*{ij})}{\sum*{k \in \mathcal{N}_i} \exp(e_{ik})} $$

### 4. Aggregation

Finally, we compute the output features for node $i$ as a weighted sum of the neighbors' features, followed by a nonlinearity $\sigma$ (usually ELU or ReLU).
$$ \vec{h}'_i = \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}\vec{h}\_j \right) $$

### 5. Multi-Head Attention

To stabilize learning, GATs use multi-head attention (similar to Transformers). We execute $K$ independent attention mechanisms and concatenate (or average) their results.

- **Concatenation** (Hidden layers): $\vec{h}'_i = \|_{k=1}^K \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} \mathbf{W}^{(k)}\vec{h}_j \right)$
- **Averaging** (Output layer): $\vec{h}'_i = \sigma \left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} \mathbf{W}^{(k)}\vec{h}_j \right)$

---

## PyTorch Implementation

We will implement a GAT layer from scratch using pure PyTorch. This implementation assumes we have the full adjacency matrix (dense or sparse). For efficiency in large graphs, one would typically use sparse matrix operations or scatter/gather operations (like in PyTorch Geometric), but here we focus on clarity and understanding the math.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate (True) or average (False) activation
                    (Concatenation is usually for hidden layers, averaging for output)
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # W: Learnable weight matrix (F x F')
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # a: Learnable attention vector (2F' x 1)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        Args:
            h: Input node features (N x in_features)
            adj: Adjacency matrix (N x N)
        """
        # 1. Linear Transformation
        # h: (N, in_features), W: (in_features, out_features) -> Wh: (N, out_features)
        Wh = torch.mm(h, self.W)
        N = Wh.size()[0]

        # 2. Attention Mechanism
        # We need to compute a^T [Wh_i || Wh_j] for all pairs (i, j).
        # A clever way to vectorize this:
        # a_input = [Wh_i || Wh_j]
        # But we can split a into a1 and a2 (both size out_features x 1)
        # a^T [Wh_i || Wh_j] = a1^T Wh_i + a2^T Wh_j

        a1 = self.a[:self.out_features, :] # (out_features, 1)
        a2 = self.a[self.out_features:, :] # (out_features, 1)

        # (N, out) x (out, 1) -> (N, 1)
        e1 = torch.matmul(Wh, a1)
        e2 = torch.matmul(Wh, a2)

        # Broadcast add to get (N, N) matrix where [i, j] is e1[i] + e2[j]
        # e: (N, N)
        e = e1 + e2.T

        e = self.leakyrelu(e)

        # 3. Masked Attention (use adjacency to only attend to neighbors)
        # We assume adj is 1 for edges and 0 otherwise.
        # We set non-neighbors to -infinity so softmax makes them 0.
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # 4. Softmax Normalization
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 5. Aggregation
        # h_prime = attention * Wh
        # (N, N) x (N, out) -> (N, out)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention for the first layer
        self.attentions = [
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]

        # Register them as sub-modules so they show up in model.parameters()
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # Output layer (single head or averaged multi-head, here we do simple single head for classification)
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        # Concatenate outputs of all heads
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)

        # Output layer
        x = F.elu(self.out_att(x, adj))

        # Log softmax for classification
        return F.log_softmax(x, dim=1)
```

```python
# Usage Example

# Dummy data
N = 5  # Number of nodes
F_in = 10 # Input features
F_out = 2 # Number of classes

x = torch.randn(N, F_in)

# Random adjacency matrix (binary)
adj = torch.randint(0, 2, (N, N)).float()
# Add self-loops
adj = adj + torch.eye(N)
adj[adj > 1] = 1

model = GAT(nfeat=F_in, nhid=8, nclass=F_out, dropout=0.6, alpha=0.2, nheads=2)

output = model(x, adj)
print("Output shape:", output.shape)
print("Output probabilities:\n", torch.exp(output))
```

This implementation highlights the core mechanics:

1.  **Broadcasting** to compute all pairs efficiently.
2.  **Masking** with the adjacency matrix to respect graph structure.
3.  **Multi-head attention** logic.

## Notebook

Check the [notebook](https://gist.github.com/AhmedCoolProjects/c0247e0418e9227ef5787830e479614b)
