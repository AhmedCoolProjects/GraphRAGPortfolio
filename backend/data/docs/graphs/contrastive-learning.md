# Contrastive Learning

> **Note**: This guide covers **Contrastive Learning**, which is the standard self-supervised counterpart to Generative Learning (like MAE).

Contrastive Learning is a powerful paradigm in Self-Supervised Learning (SSL) that learns representations by **contrasting** positive pairs against negative pairs. Instead of trying to reconstruct the input (like AutoEncoders), it tries to learn an embedding space where similar samples are close and dissimilar samples are far apart.

## The Concept

The core intuition is simple:

1.  **Positive Pair**: Two augmented views of the _same_ graph (or node) should have similar representations.
2.  **Negative Pair**: Views of _different_ graphs (or nodes) should have different representations.

By forcing the model to distinguish between "versions of itself" and "others", the model learns robust, discriminative features that are invariant to the applied augmentations (noise).

### Contrastive vs. Generative (MAE)

| Feature           | Contrastive Learning (e.g., GraphCL)  | Generative Learning (e.g., GraphMAE)   |
| :---------------- | :------------------------------------ | :------------------------------------- |
| **Objective**     | Discrimination (Who is who?)          | Reconstruction (What was here?)        |
| **Input**         | Multiple Augmented Views              | Masked Input                           |
| **Loss Function** | InfoNCE (Cross-Entropy)               | MSE (Reconstruction Error)             |
| **Pros**          | Learns very robust global features    | Captures fine-grained local details    |
| **Cons**          | Relies heavily on "Negative Sampling" | Computationally cheaper (no negatives) |

## Examples in Graphs

1.  **GraphCL (Graph Contrastive Learning)**:
    - Applies augmentations like _Node Dropping_, _Edge Perturbation_, _Subgraph Extraction_, and _Feature Masking_.
    - Maximizes agreement between two augmented views of the same graph.
2.  **DGI (Deep Graph Infomax)**:
    - Contrasts node representations (local) with the graph summary (global).
    - Maximizes mutual information between the node and the graph it belongs to.
3.  **GRACE / GCA**:
    - Node-level contrastive learning.
    - Contrasts a node with itself in another view vs. all other nodes.

## Power and Weaknesses

### Power

- **Discriminative Features**: Excellent for classification tasks where distinguishing classes is key.
- **No Labels Needed**: Can pre-train on massive unlabeled datasets.
- **Invariance**: Explicitly encodes invariance to noise (e.g., if you train it to ignore "Edge Perturbation", the model becomes robust to missing edges).

### Weaknesses

- **Augmentation Sensitivity**: The success depends heavily on _choosing the right augmentations_. For example, dropping edges in a molecule might change its chemical property, making it a "false positive".
- **Negative Sampling**: Requires a large number of negative samples (large batch sizes) to work well.
- **Sampling Bias**: Random negatives might actually be semantically similar (False Negatives), confusing the model.

## The Math: InfoNCE Loss

The standard loss function is **InfoNCE** (Information Noise Contrastive Estimation).

Let $z_i$ and $z_j$ be the representations of two augmented views of the same graph (Positive Pair).
Let $\{z_k\}$ be the set of representations of other graphs in the batch (Negatives).

The similarity is usually Cosine Similarity:
$$ \text{sim}(z_i, z_j) = \frac{z_i^T z_j}{\|z_i\| \|z_j\|} $$

The loss for a single pair $(i, j)$ is:

$$ \mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}\_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)} $$

Where:

- $\tau$ is a temperature parameter (controls how sharp the distribution is).
- The numerator pulls positive pairs together.
- The denominator pushes the positive pair away from all other $2N-1$ samples (negatives).

## PyTorch Implementation

Here is a clean implementation of **GraphCL** logic using PyTorch. We simulate the augmentations and the contrastive loop.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphEncoder(nn.Module):
    """Simple GCN Encoder"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, 64) # Simulating GCN layer
        self.conv2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim)
        # Projection head: maps representations to space where contrastive loss is applied
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

def contrastive_loss(z1, z2, temperature=0.5):
    """
    Computes InfoNCE loss.
    z1, z2: (Batch_Size, Dim) - Projections of View 1 and View 2
    """
    batch_size = z1.shape[0]

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate all representations: [z1; z2] -> (2*Batch, Dim)
    z = torch.cat([z1, z2], dim=0)

    # Compute similarity matrix: (2B, 2B)
    sim_matrix = torch.matmul(z, z.T) / temperature

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, -9e15)

    # For each sample in z1, its positive is in z2 (and vice versa)
    # Indices: 0 matches with B, 1 matches with B+1, ...

    # We want to maximize sim(z[i], z[i+B])

    # Create labels for CrossEntropy
    # For i in 0..B-1 (z1), target is i+B
    # For i in B..2B-1 (z2), target is i-B
    labels_1 = torch.arange(batch_size, device=z.device) + batch_size
    labels_2 = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels_1, labels_2], dim=0)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# --- Simulation of Training Loop ---
if __name__ == "__main__":
    # Hyperparameters
    B = 32 # Batch size
    D = 16 # Input dimension
    H = 8  # Hidden dimension

    model = ContrastiveModel(D, H)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Dummy Data (Batch of graphs represented as feature vectors for simplicity)
    # In real GraphCL, x would be graph objects, and we'd apply augmentations here.
    x = torch.randn(B, D)

    # Simulating Augmentations:
    # View 1: Add some noise
    view1 = x + 0.1 * torch.randn_like(x)
    # View 2: Mask some features
    view2 = x * (torch.rand_like(x) > 0.2).float()

    # Forward
    _, z1 = model(view1)
    _, z2 = model(view2)

    # Loss
    loss = contrastive_loss(z1, z2, temperature=0.5)

    print(f"Contrastive Loss: {loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This code demonstrates the "SimCLR" style framework which is the backbone of most Graph Contrastive Learning methods. The key is in the `contrastive_loss` function which efficiently computes similarities between all pairs in the batch.
