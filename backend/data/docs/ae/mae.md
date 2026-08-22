# Masked AutoEncoders (MAE)

Masked AutoEncoders (MAE) are scalable self-supervised learners for computer vision, introduced by He et al. (2021). They revolutionize how we approach representation learning on images by demonstrating that masking a high proportion (e.g., 75%) of the input image and reconstructing the missing pixels is a powerful pre-training task.

## MAE vs. Standard AutoEncoders

In a **Standard AutoEncoder (AE)**, the encoder maps the _entire_ input to a latent representation, and the decoder attempts to reconstruct the _entire_ input from that latent code. The bottleneck is usually the dimension of the latent space.

In **MAE**, the process is asymmetric:

1.  **Masking**: We mask a large portion of the input image patches.
2.  **Encoder**: Only the _visible_ (unmasked) patches are fed into the encoder. This saves significant computation and memory.
3.  **Decoder**: The decoder takes the latent representations of the visible patches _and_ learnable "mask tokens" for the missing patches to reconstruct the original image.

| Feature                      | Standard AutoEncoder    | Masked AutoEncoder (MAE)                   |
| :--------------------------- | :---------------------- | :----------------------------------------- |
| **Input to Encoder**         | Full image              | Only visible patches (~25%)                |
| **Encoder/Decoder Symmetry** | Usually symmetric       | Asymmetric (Heavy Encoder, Light Decoder)  |
| **Objective**                | Compression / Denoising | Inpainting / Understanding Context         |
| **Efficiency**               | Processes full input    | Encoder processes only a fraction of input |

## Relationship with Masked Language Modeling (MLM)

MAE is conceptually similar to **BERT** (Masked Language Modeling) in NLP.

- **BERT**: Masks tokens in a sentence and predicts the missing words.
- **MAE**: Masks patches in an image and predicts the missing pixel values.

**Key Differences**:

- **Density**: Language is information-dense; masking a few words makes the task hard. Images have high spatial redundancy; we must mask a _very high_ percentage (e.g., 75%) to create a challenging task that forces the model to learn high-level semantics rather than low-level interpolation.
- **Decoder Role**: In BERT, the encoder does all the heavy lifting (reconstruction happens at the end). In MAE, a separate decoder is crucial for reconstructing pixels, while the encoder focuses purely on latent representation.

## The Forward Pass

1.  **Patchify**: Divide the image ($H \times W \times C$) into non-overlapping patches ($P \times P$).
2.  **Masking**: Randomly sample a subset of patches to keep (e.g., 25%) and discard the rest.
3.  **Encoding**:
    - Flatten the visible patches and project them to embedding dimension $D$.
    - Add positional embeddings.
    - Pass through a Transformer Encoder.
4.  **Decoding**:
    - Append learnable **Mask Tokens** to the sequence of encoded visible patches to restore the original sequence length.
    - Add positional embeddings (again) to all tokens.
    - Pass through a Transformer Decoder (typically smaller/shallower than the encoder).
5.  **Reconstruction**: A linear projection layer maps the decoder output back to pixel values ($P \times P \times C$).
6.  **Loss**: Compute Mean Squared Error (MSE) between the reconstructed image and the original image, _only on the masked patches_.

## The Math

Given an input image $x \in \mathbb{R}^{H \times W \times C}$.
We split it into $N = HW/P^2$ patches $x_p \in \mathbb{R}^{N \times (P^2 C)}$.

Let $\mathcal{M}$ be the set of indices for masked patches, and $\mathcal{V}$ be the set of indices for visible patches, such that $|\mathcal{V}| / N \approx 0.25$.

**Encoder**:
$$ z\_\mathcal{V} = \text{Encoder}(\text{Embed}(x_i) + \text{Pos}\_i \mid i \in \mathcal{V}) $$

**Decoder Input**:
We construct the full sequence $z_{full}$ where:
$$ z*i = \begin{cases} z_i & \text{if } i \in \mathcal{V} \text{ (from encoder)} \\ \text{Token}*{mask} & \text{if } i \in \mathcal{M} \text{ (learnable parameter)} \end{cases} $$
$$ \hat{x} = \text{Decoder}(z*{full} + \text{Pos}*{full}) $$

**Loss Function**:
$$ \mathcal{L} = \frac{1}{|\mathcal{M}|} \sum\_{i \in \mathcal{M}} (x_i - \hat{x}\_i)^2 $$
Note: The loss is typically normalized by the patch variance.

## PyTorch Implementation

Here is a simplified implementation of MAE using standard PyTorch components.

```python
import torch
import torch.nn as nn
import numpy as np

class MAE(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 mask_ratio=0.75):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_dim = patch_size * patch_size * in_chans
        self.mask_ratio = mask_ratio

        # --------------------------------------------------------------------------
        # MAE Encoder
        # --------------------------------------------------------------------------
        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Class token (optional in MAE, but often used) and Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim*mlp_ratio),
                                                   activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------
        # MAE Decoder
        # --------------------------------------------------------------------------
        # Project encoder embedding to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder Positional Embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))

        # Transformer Decoder Blocks
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=decoder_num_heads,
                                                   dim_feedforward=int(decoder_embed_dim*mlp_ratio),
                                                   activation='gelu', batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Prediction Head
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (sin-cos) pos_embed and other weights...
        # For brevity, using simple normal initialization here
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.decoder_pos_embed, std=.02)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x) # [N, C, H, W] -> [N, Embed, H/P, W/P]
        x = x.flatten(2).transpose(1, 2) # [N, Embed, L] -> [N, L, Embed]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.encoder(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)

        # Mean Squared Error
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # Apply mask: calculate loss only on masked patches
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

```

```python
model = MAE(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16)

img = torch.randn(2, 3, 224, 224)
loss, pred, mask = model(img)
print(f"Loss: {loss.item()}")
print(f"Prediction shape: {pred.shape}")
```

## Notebook

Check the [notebook](https://gist.github.com/AhmedCoolProjects/6ce9ae11dac1f46bed689d3a956c5856).
