# Positional Encoding

Since the Transformer processes all words in **parallel** _(simultaneously)_ rather than sequentially like an RNN, it has no inherent sense of **order**.

Without **Positional Encoding**, the Transformer would see "The dog bit the man" and "The man bit the dog" as the exact same bag of words.

**The "Timestamp" Solution**

To fix this, we literally **add** a vector of information to each word embedding before it enters the encoder. This vector acts like a **timestamp** or a unique ID card that say, "I am the 1st word", "I am the 5th word", etc.

## How it works

We might think to just add integers _(e.g., 1 for first word, 2 for second, etc.)_. But this is not a good idea because it would give the model the false impression that 2 is twice as important as 1, or that 3 is three times as important as 1.

The authors of the Transformer paper came up with a clever solution using **sine and cosine functions** of different frequencies.

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

This creates a unique, continuous pattern for every position that the model can easily learn to attend to.

- **Low frequencies** change slowly _(like the hour hand on a clock)_
- **High frequencies** change quickly _(like the second hand on a clock)_

By combining these, every position gets a unique "fingerprint" vector.

**Sine and Cosine are Perfect because**:

1. **Bounded**: They always stay between **-1 and 1**, keeping the math stable no matter how long the sentence is.
2. **Relative Position**: The math of waves makes it easy for the model to learn relative distances _(like "pay attention to the word 3 steps behind me") because the wave pattern shifts predictably.

## Understanding the Formula

Let's break it down. We aren't just creating a single number for each position; we are creating a **whole vector** of numbers for each position.

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

Let's look at the variables:
- $pos$: The position of the word in the sentence _(e.g., 0 for first word, 1 for second)_.
- $i$: The index of the dimension in the vector _(e.g., if our embedding size $d_{model}$ is 512, $i$ goes from 0 to 255)_.
- The Denominator $10000^{2i/d_{model}}$: This determines the **frequency** _(or speed)_ of the wave.

**The "Many Hands of a Clock" Analogy**

Let's imagine the positional encoding vector is a row of **clocks**.

1. **Low $i$ (Start of the vector)**: The denominator is small $10000^0 = 1$. The wave oscillates **very fast**.
    - This is like the **second hand** on a clock. It spins around wildly as you move from word 1 to word 2 to word 3.
2. **High $i$ (End of the vector)**: The denominator is large $10000^1 = 10000$. The wave oscillates **very slowly**.
    - This is like the **hour hand**. It barely moves as you go from word 1 to word 2. It takes a thousands of words for it to complete on cycle.

**How this creates a Unique "Fingerprint"**

For any specific word position _(say, $pos = 5$)_, we take a snapshot of all these clocks at once.
- Clock 1 _(High Frequency)_ might be pointing at 3 o'clock.
- Clock 2 _(Medium Frequency)_ might be pointing at 12 o'clock.
- Clock 3 _(Low Frequency)_ might be pointing at 12:01.

That specific combination of "[3:00, 12:00, 12:01]" is the unique ID for Position 5. No other position will have that exact combination.

- **Position 6** will have a very different "Clock 1" _(because it spins faster)_, but a nearly identical "Clock 3".
- **Position 1000** will have a totally different "Clock 3".

**Why Sine and Cosine?**

We use pairs of $\sin$ and $\cos$ for a beautiful mathematical reason relating to **relative positions**.

In trigonometry, there is a formula: $\sin(x + k) = \sin(x)\cos(k) + \cos(x)\sin(k)$

This means we can express the encoding for position $pos + k$ _(a word k steps away)_ as a **linear function** of the encoding for position $pos$.

This makes it very easy for the model's "Self-Attention" layers _(which do linear matrix multiplications)_ to learn relationships like "attend to the word 3 steps behind me" regardless of whether we are at index 5 or index 500.


## Python Implementation

```python title="Positional Encoding"
import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        '''
        d_model: The embedding dimension (e.g., 512)
        max_len: The maximum possible length of a sequence we expect to process
        '''
        # 1. Initialize the matrix with zeros
        # Shape (max_len, d_model)
        pe = np.zeros((max_len, d_model))

        # 2. Create the position indices (0, 1, ..., max_len-1)
        pos = np.arange(0, max_len).reshape(-1, 1)

        # 3. Calculate the "div_term" (for frequencies)
        # This implements: 1 / 10000^(2i/d_model)
        # We use a log-space trick for numerical stability
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        # 4. Fill the matrix
        # Even indices (2i) get Sine
        pe[:, 0::2] = np.sin(pos * div_term)

        # Odd indices (2i+1) get Cosine
        pe[:, 1::2] = np.cos(pos * div_term)

        self.pe = pe # shape (max_len, d_model)
    
    def forward(self, x):
        '''
        x: Input embeddings of shape (batch_size, seq_len, d_model)
        '''
        # We slice 'pe' to match the actual sequence length of 'x'
        # and ADD it to the input embeddings
        x = x + self.pe[:x.shape[1], :] # shape (batch_size, seq_len, d_model)
        return x
```