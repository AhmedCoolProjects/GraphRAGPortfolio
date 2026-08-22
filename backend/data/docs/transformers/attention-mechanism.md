# Self-Attention Mechanisms

The core idea of **self-attention** is that it allows the model to look at every word in a sentence *simultaneously* and decide how much **attention** to pay to each one when trying to understand a specific word.

To do this, we create three distinct vectors _(lists of numbers)_ for each word.

1. **Query _($Q$)_**: This represents the word asking for information. _(e.g., "I am 'bank', what kind of bank am I?")_
2. **Key _($K$)_**: This represents the label or tag on every other word. It helps identify if that word is relevant. _(e.g., "I am 'river', I'm about nature/water.")
3. **Value _($V$)_**: This is the actual content or meaning of the word that we will extract if it's a match.

## Example

Let's take an example:

> "The bank of the river"
> 
> The word 'bank' is asking for context: The Query _($Q$)_ is the one looking for information now.
> 
> The Query _($Q$)_ looks at the **Keys** _($K$)_ of every other word _('the', 'of', 'river')_ to see which ones are relevant.
> 
> The 'river' word provides the crucial context that disambiguates 'bank'. Without 'river', 'bank' could easily be a financial institution.

### Attention Score

How the Match Works? _(**Dot Product**)_

In the model, this **matching** isn't magic, it's geometry:
- We take the **Query** _($Q$)_ and **Key** _($K$)_ vectors and calculate the **dot product** between them.
- The dot product is a measure of similarity between the two vectors. Since 'bank' and 'river' are semantically related in the model's training, their vectors align, producing a **high score**.
- Irrelevant words like 'the' or 'of' produce a **low score**.

Now we have a list of scores _(e.g., 'river': high, 'the': low)_. We apply a **softmax** function to these scores to convert them into **probabilities** so they sum to 1.

The score we calculate is called the **Attention Score**.

### Context Vector

When we have the **attention scores** of all other words, we calculate the **context vector** which is a weighted sum of the **Values** _($V$)_ of all other words based on their **attention scores**.

$$
Z_{bank} = \sum_{i=1}^{n} \text{Attention Score}(Q_{bank}, K_{i}) \times V_{i}
$$

**The Result** is the new representation for 'bank' _($Z_{bank}$)_. This is how the model understands context! The word 'bank' has absorbed the meaning of its neighbors.

### Multi-Head Attention

**Problem of Multiple Relationships**

Now let's imagine a slightly more complex sentence:

> "The bank of the river gave a loan".
>
> Here, 'bank' has two important relationships:
> 1. **river** _(physical location context)_
> 2. **loan** _(financial context)_
>

If we only have **one** set of $Q, K, V$ for 'bank' _(one **attention head**)_ , the model will have difficulty distinguishing between these two relationships, the model has to choose: should the new 'bank' vector look more like 'river' or more like 'loan'? It might get a muddy average of both.

To solve this problem, we use **multi-head attention**. Just like having multiple eyes to see different things in the same scene, each **head** learns to focus on a different types of relationships.
- **Head 1** might learn **semantic** relationships _(linking 'bank' to 'river')_
- **Head 2** might learn **grammatical** relationships _(linking 'gave' to the subject 'bank')_
- **Head 3** might focus on **punctuation** or rare words.

**How it works mechanically**

1. **Split**: Instead of one giant set of $Q, K, V$ matrices, we create multiple smaller sets.
2. **Parallel Processing**: Each head performs the attention calculation independently. Head 1 outputs $Z_1$, Head 2 outputs $Z_2$, etc.
3. **Concatenate**: We stitch all these output vectors _($Z_1, Z_2, ...$)_ together into one long vector.
4. **Final Linear Layer**: We multiply this long vector by a final weight matrix $W_o$ to compress it back down to the standard size.

This gives us a final vector for 'bank' that captures **all** its different contexts simultaneously.

## Limitations of Self-Attention

After all this, we have a powerful mechanism that connects words based on meaning. But there is a major flaw.

If we scrambled the sentence to say 'River the of bank the', the self-attention mechanism _(which just compares every word pair)_ would calculate the **exact same attention scores**. It has no inherent concept of **first**, **second** or **next**.

This is where **Positional Encoding** comes in. Check it [here](./positional-encoding)

## Python Implementation

**The Strategy: Matrix Efficiency**

Instead of creating 8 separate small matrices for 8 heads _(which would be slow)_, we use a standard optimization trick:
1. **One Big Matrix**: We create one giant $W_Q$, $W_K$, $W_V$ of size $d_{model}$.
2. **Split via Reshape**: After multiplying, we assume the resulting vector is actually 8 smaller vectors glued together. We use $reshape$ and $transpose$ to split them apart.

```python title="Multi-Head Self-Attention"
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, batch_size):
        '''
        d_model: The embedding dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        '''
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d = d_model
        self.n = num_heads
        self.m = batch_size
        self.d_k = d_model // num_heads # Dimension of each head (e.g., 64)
        
        # 1. Initialize Weights
        # 3 Large matrices, each (d_model, d_model)
        _shape = (d_model, d_model)
        self.W_q = np.random.randn(*_shape) * 0.01
        self.W_k = np.random.randn(*_shape) * 0.01
        self.W_v = np.random.randn(*_shape) * 0.01
        # The final output linear layer (d_model, d_model)
        self.W_o = np.random.randn(*_shape) * 0.01

    def softmax(self, z):
        e_x = np.exp(z - np.max(z, axis=-1, keepdims=True)) # axis=-1 means we are operating on the last dimension
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def split_heads(self, x):
        '''
        Splits the last dimension into (num_heads, d_k)
        Transposes the result so that 'num_heads' is the 2nd dimension
        '''
        # 1. Reshape so we can split
        # from (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, d_k)
        x = x.reshape(self.m, -1, self.n, self.d_k) # -1 means we let numpy figure it out
        # 2. Transpose so 'num_heads' is the 2nd dimension
        return x.transpose(0, 2, 1, 3)

    def concat_heads(self, x):
        # 1. Transpose back so 'seq_len' is the 2nd dimension
        x = x.transpose(0, 2, 1, 3)
        # 2. Reshape so we can concat
        # from (batch_size, num_heads, seq_len, d_k) to (batch_size, seq_len, d_model)
        return x.reshape(self.m, -1, self.d)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        '''
        Calculates the attention scores.
        Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        '''

        d_k = Q.shape[-1]

        # --- 1. MatMul Q and K^T
        # We swap the last two dimensions of K to transpose it
        # (batch, heads, seq_len, d_k) @ (batch, heads, d_k, seq_len)
        # Result: (batch, heads, seq_len, seq_len) -> The "Score Map"
        scores = np.matmul(Q, K.swapaxes(-2, -1))

        # --- 2. Scale the scores
        scores = scores / np.sqrt(d_k) # we use the square root of d_k to scale the scores because the dot product can become very large

        # --- 3. Mask (Optional)
        if mask is not None:
            # We assume mask has 0 for positions to hide, 1 for positions to keep
            # We replace 0s with -inf (or a very large negative number)
            # so that softmax makes them zero
            scores = np.where(mask == 0, -1e9, scores)

        # --- 4. Apply Softmax
        # We get probabilities (0 to 1) for each position
        attention_weights = self.softmax(scores)

        # --- 5. MatMul with V
        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, d_k)
        # Result: (batch, heads, seq_len, d_k)
        context = np.matmul(attention_weights, V)

        return context, attention_weights

    def forward(self, q, k, v, mask=None):
        '''
        The main forward pass.
        For Self-attention: q, k, v are all the same input 'x'
        For Cross-attention: q is from the decoder, k/v are from the encoder
        '''
        # 1. Linear projections (for all heads at once)
        # Result: shpae (m, s, d)
        Q = np.dot(q, self.W_q)
        K = np.dot(k, self.W_k)
        V = np.dot(v, self.W_v)

        # 2. Split heads
        # Result shape: (m, h, s, d_k)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Calculate attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. Concatenate heads
        # Result shape: (m, s, d)
        concat_context = self.concat_heads(context)

        # 5. Final linear projection
        # Result shape: (m, s, d)
        output = np.dot(concat_context, self.W_o)

        return output, attention_weights
```

**Quick Test**

```python title="Quick Test
import numpy as np

d_model = 512
num_heads = 8
batch_size = 2
seq_len = 10

# Create the layer
mha = MultiHeadAttention(d_model, num_heads, batch_size)

# Generate fake input data
X = np.random.randn(batch_size, seq_len, d_model)

# Run the forward pass
output, attention_weights = mha.forward(X, X, X)

print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)
print("Test Passed!")
```