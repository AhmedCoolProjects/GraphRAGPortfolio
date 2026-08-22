# Modern BERT

## Absolute vs. Relative

To understand the difference between absolute and relative positional encodings, we have to look at the **Self-Attention** mechanism, which is the heart of the Transformer.

Recall that the core job of attention is to calculate a **score** between two words (Query $Q$ and Key $K$) to decide how much they should "talk" to each other.

$$\text{Score} = Q \cdot K^T$$

#### 1. The "Absolute" Way (Original BERT)

In the original [BERT](./bert), we treat position like a **house address**.

- We learn a static vector for "Position 1," another for "Position 2," and so on.
- We simply **add** this position vector to the word embedding *before* it enters the Transformer.

    $$\text{Input} = Word_{Embedding} + Position_{Embedding}$$



**The Limitation:**
The model has to *learn* the relationship between every pair of positions from scratch. It doesn't inherently know that "Position 1" is close to "Position 2." It just sees two different vectors. If you train on sequences of length 512, the model has **no address book** for "Position 513." It literally cannot see past that point.

#### 2. The "Relative" Way (RoPE in ModernBERT)

[cite_start]ModernBERT uses **Rotary Positional Embeddings (RoPE)**[cite: 1019], which treats position like a **distance** or a relationship.

Instead of adding a static vector at the start, RoPE modifies the $Q$ and $K$ vectors *inside* the attention layer by **rotating** them.

- Imagine the vector is an arrow on a graph.
- If a word is at **Position $m$**, we rotate its vector by angle $m\theta$.
- If a word is at **Position $n$**, we rotate its vector by angle $n\theta$.

**The Magic:**
When we calculate the dot product ($Q \cdot K^T$) to get the attention score, the geometry of rotation means the result depends **only on the angle difference** $(m - n)$.

$$Q_m \cdot K_n \propto \cos(m - n)\theta$$

This means the model no longer cares "where" the words are globally (index 5 vs. index 10). It only cares that they are **5 steps apart** ($10 - 5 = 5$).

### Why this matters for ModernBERT

By using RoPE, ModernBERT achieves two major things:
1.  **Generalization:** It understands "nearness" naturally. [cite_start]If it learns that "not" affects the word immediately following it (distance +1), it applies that logic everywhere, even at sequence lengths it hasn't seen much[cite: 1020].
2.  [cite_start]**Long Context:** It allows the model to extend its native context length to **8,192 tokens** [cite: 983, 1085] because it doesn't need to learn a new static vector for position 8,000; it just calculates the rotation.

[cite_start]Based on this, why do you think this "relative" approach (caring about distance, not address) helps the model handle **code** [cite: 1004] better than the original BERT? (Think about how code structure works compared to simple sentences).

## Positional Encoding (RoPE)

ModernBERT moves away from the absolute positional embeddings used in the original BERT model and instead adopts **Rotary Positional Embeddings** _(RoPE)_.

Here is a breakdown of why this change matters:

- **The Old Way (Absolute)**: Original BERT models assigned a unique, fixed vector to each position (e.g., Position 1, Position 2). This worked fine for short sequences but struggled to generalize to lengths it hadn't seen during training.
- **The Modern Way (RoPE)**: RoPE encodes position information by rotating the token embeddings in the vector space. The amount of rotation depends on the position of the token.
- **The Benefit**: This method allows the model to understand the relative distance between tokens much better than absolute positions do. It is also highly efficient and makes it easier to extend the model's context length to $8,192$ tokens and beyond.