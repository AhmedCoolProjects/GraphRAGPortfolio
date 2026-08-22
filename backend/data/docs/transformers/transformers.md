# Transformers

## Python Implementation

```python title="Transformer"
import numpy as np

class Transformer:
    def __init__(self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        dropout=0.1
        ):

        self.d = d_model

        # 1. Encoder Stack
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, pe_input)

        # 2. Decoder
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, pe_target)

        # 3. Final Output Layer
        self.final_layer = np.random.randn(d_model, target_vocab_size) * 0.01
        self.final_bias = np.zeros((1, target_vocab_size))

    def create_padding_mask(self, seq):
        """
        Creates a mask where 1 indicates padding (0) and 0 indicates real token.
        seq shape: (batch_size, seq_len)
        Output shape: (batch_size, 1, 1, seq_len) for broadcasting
        """
        seq = (seq == 0).astype(float)
        return seq[:, np.newaxis, np.newaxis, :]

    def create_look_ahead_mask(self, size):
        """
        Creates a triangular mask to hide future tokens.
        Output shape: (size, size)
        """
        mask = np.triu(np.ones((size, size)), k=1)
        return mask # 1 means hide, 0 means keep

    def create_masks(self, inp, tar):
        """
        Generates all necessary masks for the forward pass.
        """
        # 1. Encoder Padding Mask
        # Prevents encoder from attending to padding tokens in the input
        enc_padding_mask = self.create_padding_mask(inp)
        
        # 2. Decoder Padding Mask (for Cross-Attention)
        # Prevents decoder from attending to padding tokens in the encoder output
        dec_padding_mask = self.create_padding_mask(inp)
        
        # 3. Look-Ahead Mask (for Self-Attention in Decoder)
        # Used so decoder can't see future targets
        look_ahead_mask = self.create_look_ahead_mask(tar.shape[1])
        
        # 4. Decoder Target Padding Mask
        # Prevents decoder from attending to padding in the target sequence itself
        dec_target_padding_mask = self.create_padding_mask(tar)
        
        # Combine look-ahead and padding mask for the first attention block
        # If EITHER is 1 (hide), we want to hide it.
        combined_mask = np.maximum(look_ahead_mask, dec_target_padding_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask

    def forward(self, inp, tar):
        # 1. Masks
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        # 2. Run Encoder
        enc_output = self.encoder.forward(inp, enc_padding_mask)

        # 3. Run Decoder
        dec_output = self.decoder.forward(tar, enc_output, look_ahead_mask, dec_padding_mask)

        # 4. Final Linear Layer
        final_output = np.dot(dec_output, self.final_layer) + self.final_bias

        # Note: We usually return logits (raw scores) and apply softmax during loss calculation
        return final_output
```

```python title="Quick Test"
# Hyperparameters
num_layers = 4
d_model = 128
d_ff = 512
num_heads = 8
input_vocab_size = 5000
target_vocab_size = 5000
dropout_rate = 0.1

# Create Model
transformer = Transformer(num_layers, d_model, num_heads, d_ff, 
                            input_vocab_size, target_vocab_size, 
                            pe_input=1000, pe_target=1000)

# Fake Data (Batch of 2 sentences)
# Input: [Start, word1, word2, Pad, Pad]
sample_input = np.array([[1, 24, 55, 0, 0], 
                            [1, 88, 92, 12, 0]])

# Target: [Start, word1, word2, word3, Pad]
sample_target = np.array([[1, 44, 22, 90, 0], 
                            [1, 32, 55, 11, 0]])

print("Input Shape:", sample_input.shape)   # (2, 5)
print("Target Shape:", sample_target.shape) # (2, 5)

# Run Forward Pass
# This will generate masks internally and run the whole flow
fn_out = transformer.forward(sample_input, sample_target)

print("\n--- Transformer Output ---")
print("Output Shape:", fn_out.shape) # Expected: (2, 5, 5000)
print("(Batch Size, Target Seq Len, Target Vocab Size)")
print("\nTest Passed! The Transformer successfully processed the sequence.")
```