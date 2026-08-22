# BERT

**Bidirectional Encoder Representations from Transformers** is essentially a transformer Encoder. Unlike the original transformer _(which has an Encoder and a Decoder)_ or GPT _(which is a decoder-only model)_, BERT relies entirely on the Encoder mechanism to understand text.

## Masked Language Model

**MLM** changes the rule of language models from *left-to-right* to *fill-in-the-blank* _(often called Cloze test)_.

- **Input**: we feed the model the sentence "The man went to the bank" but we mask some words by replacing them with a special token `[MASK]`. This masking is done randomly.
- **Goal**: the model has to look at the context _(words surrounding the mask)_ to predict the original word.

As an example of an input: "The [MASK] went to the bank"

By hiding words in the middle, we force the model to look at the words before and after simultaneously, this creates a deep, bidirectional understanding of the sentence structure and meaning.

## Masked Word Prediction

To understand how this is a breakthrough, we need to look at how models read text before BERT.

Most earlier models like standard LSTMs read text in one direction _(left-to-right)_.

As an example, let's take the following sentence:

> "The man went to the bank to catch some fish"

In this case, the model would first process "The", then "man", then "went", and so on. It doesn't have any context about the future words in the sentence. This way when it processes "bank", it would not know if it is a bank _(financial institution)_ or a bank _(river bank)_.

BERT solves this issue by using a Transformer Encoder, which allows it to see the entire sentence at once but with a masked language model to train it for better understanding of the language.

## Next Sentence Prediction

BERT is not just trained on words; it is also trained to understand the relationships between sentences. This is crucial for tasks like _question answering_ or _text classification_.

To learn this, BERT plays another game called **Next Sentence Prediction**. We feed the model a pair of sentences and ask it to predict if the second sentence is the next sentence in the story or not.

- **50%**: the second sentence is the next sentence in the story.
- **50%**: the second sentence is a random sentence from another story.

The model has to output a Yes/No answer.

## Input Format

Now, here is the tricky part, we need to feed the model a pair of sentences at the same time. But BERT acts like a single long Encoder. It takes a stream of words.

> "The man went to the shop he bought milk.", in this case, the model will see the entire sentence at once but it will not know where's the end of the first sentence or the start of the second sentence.

To solve this issue, BERT uses a special token `[SEP]` to separate the two sentences.

> "The man went to the shop [SEP] he bought milk."

Let's say the task of the model is to output if the sentence is happy or sad _(Classification Task)_. We can average the vectors of the set of tokens, but that would be messy. For that, BERT uses a special token `[CLS]` _(Classification Token)_ at the beginning of the sentence. This token is used to represent the entire sentence and its context. 

So every time we want to perform a classification task, we will feed the model the input and get the output of the `[CLS]` token. 

> "[CLS] The man went to the shop [SEP] he bought milk."

## Pre-Training vs. Fine-Tuning

We covered now two stages of BERT's life: pre-training and fine-tuning. 

- **Pre-training**: the model reads massive amounts of text _(Wikipedia, BooksCorpus, etc.)_ and plays the _MLM_ and _NSP_ games to learn the grammar and context of the language. This takes weeks on massive GPUs.
- **Fine-tuning**: we take this **smart** pre-trained model and add a small classifier on top for `[CLS]` token. We then train it on a specific task _(like Sentiment Analysis)_. 

## BERT Embeddings

BERT is a transformer encoder with a masked language model and next sentence prediction tasks. Thus, it has the same architecture as the original transformer encoder, but with an extra layer after the `Word Embedding` and `Position Embedding` layers. This layer is called `Token Type Embedding` _(or `Segment Embedding`)_.

To fully represent a pair of sentences _("My dog", "He barks")_, BERT sums up 3 vectors for each token:
1. **Token Embedding**: What's the word?
2. **Position Embedding**: Where is it?
3. **Segment Embedding**: In which sentence is it?

**How Segment Embedding works?**

It is quite simple, each token gets added to a vector representing the first sentence or the second sentence.

> Tokens: `[CLS]`, `My`, `dog`, `[SEP]`, `He`, `barks`, `[SEP]`
>
> Segment Embeddings: `0`, `0`, `0`, `0`, `1`, `1`, `1`

## Model Architecture

BERT comes in two main sizes:

- **BERT Base**: 12 Layers (Transformer Blocks), 768 Hidden Size, 12 Attention Heads, 110M Parameters.
- **BERT Large**: 24 Layers (Transformer Blocks), 1024 Hidden Size, 16 Attention Heads, 340M Parameters.

## Tokenization

BERT uses **WordPiece** tokenization. It breaks down words into smaller subwords (tokens).

- Common words remain as whole words: `play`, `run`.
- Rare words are split into subwords: `playing` -> `play`, `##ing`. The `##` indicates that this token is part of the previous word.
- **Special Tokens**:
    - `[CLS]`: Start of every sequence. Used for classification.
    - `[SEP]`: Separator between sentences.
    - `[MASK]`: Used for Masked Language Modeling.
    - `[PAD]`: Padding token to make sequences the same length.
    - `[UNK]`: Unknown token for characters not in the vocabulary.

## Lifecycle of BERT

### 1. Tokenization

`Raw text` -> `Integer IDs`

Before any neural network math happens, we need to convert text into numbers. BERT uses **WordPiece** tokenization.

**The Logic**: Standard word-level tokenization has a problem: the vocabulary is __infinite__. If you train on "play", "playing", and "played", you treat them as three totally unrelated words. And if you see "uninstagrammable", you crash _(Out of Vocabulary)_.

**WordPiece** solves this by breaking words down into frequent sub-units.
- "playing" $\rightarrow$ ["play", "##ing"]
- "uninstagrammable" $\rightarrow$ ["un", "##insta", "##gram", "##mable"]

**The Math (Optimization)**: WordPiece is trained before BERT starts. It is an optimization problem:
- **Goal**: Maximize the likelihood of the training data using smallest possible vocabulary _(usually 30,000 tokens)_.
- **Process**: It starts with individual characters. It iteratively merges the pair of tokens _(A, B)_ that increases the likelihood of the training data the most, until the limit _(30k)_ is reached.

### 2. Input Representation

`Integer IDs` -> `Embeddings`

Now we have our list of integer IDs from WordPiece. We need to translate those boring numbers into rich, meaningful vectors.

In standard models, you usually look up one embedding for a word. BERT is different. To distinguish **what** a word is, **where** it is, and **which sentence** it belongs to, BERT looks up **three** separate vectors for every token and **sums them together**.

#### Token Embedding _($E_{tok}$)_

This is the standard dictionary lookup. ID `345` _("dog")_ points to row `345` in the lookup table _(matrix)_.

$$
E_{tok} \in \mathbb{R}^{30,522 \times 768} \text{ (30,522 is the vocabulary size, 768 is the embedding/hidden size)}
$$

#### Segment Embedding _($E_{seg}$)_

This tells the model if the token belongs to **Sentence A** or **Sentence B**. This is crucial for the Next Sentence Prediction task.

$$
E_{seg} \in \mathbb{R}^{2 \times 768} \text{ (2 is the number of segments, 768 is the embedding/hidden size)}
$$

- All tokens in Sentence A get vector `0`.
- All tokens in Sentence B get vector `1`.

#### Position Embedding _($E_{pos}$)_

Since the Transformer reads everything at once _(in parallel)_, it has no sense of order. This vector provides the "timestamp" or index _(1st, 2nd, 3rd, ...)_.

$$
E_{pos} \in \mathbb{R}^{512 \times 768} \text{ (512 is the sequence length, 768 is the embedding/hidden size)}
$$

:::note
Unlike the original Transformer which used fixed Sine/Cosine waves, BERT **learns** these position vectors from scratch during training!
:::

---
For a token $x$ at position $i$ in Sentence A:

$$
InputVector_i = LayerNorm(E_{tok}[x] + E_{seg}[0] + E_{pos}[i])
$$

We sum them **Element-wise**. They all have the exact same vector dimension _($d_{model} = 768$)_.

### 3. Forward Pass _(Encoder)_

`Contextualizing the vectors`

We have our sequence of embeddings from the previous step. Now, these vectors will pass through a stack of **Encoder Layers** _(e.g., 12 layers for BERT-Base)_.

Inside each layer, the data goes through two main sub-layers:
1. **Multi-Head Self-Attention**: It starts by computing the attention scores between all pairs of tokens in the sequence. We do have **multiple** attention heads _(8 for BERT-Base)_ to capture different types of relationships between tokens. In each head, we compute the attention scores using a **Query** and **Key**. If the Key label is what the Query is looking for, it gets a high score. If not, it gets a low score. After that, we apply the **Value** to the attention scores to get the final output which is the **contextualized** version of the meant token.

    $$
    ContextualizedVector_i = LayerNorm(\sum_{j=1}^{N} Attention(Q_i, K_j) \cdot V_j)
    $$

    where:
    - $Q_i$: Query vector for token $i$.
    - $K_j$: Key vector for token $j$.
    - $V_j$: Value vector for token $j$.
    - $Attention(Q_i, K_j)$: Attention score between token $i$ and token $j$.

2. **Position-wise Feed-Forward Network**: Since the Multi-Head Self-Attention is a Linear Weighted Sum, it cannot learn complex, non-linear patterns on its own. The **FFN** introduces that necessary non-linearity _(usually via a **ReLU** or **GELU** activation function)_, allowing the model to learn much deeper, more complex features.

    $$
    FFN(x) = ReLU(x \cdot W_1 + b_1) \cdot W_2 + b_2
    $$

    where:
    - $x$: Input vector.
    - $W_1$: First weight matrix.
    - $b_1$: First bias vector.
    - $W_2$: Second weight matrix.
    - $b_2$: Second bias vector.

So the full flow inside **one** BERT Layer is:
1. **Self-Attention**: Contextualize.
2. **Add & Norm**: Stabilize.
3. **FFN**: Process/Extract Features.
4. **Add & Norm**: Stabilize.

### 4. Pre-training _(The Loss)_

We have passed through 12 of the Encoder Layers. We now have a final vector $T_i$ for every token.

Now we need to calculate the **Loss** _(the error)_ so we can train the model. BERT minimizes **two losses** simultaneously: **MLM** (Masked Language Modeling) and **NSP** (Next Sentence Prediction).

#### MLM Loss

After the Forward Pass, we add a **MLM Head** to the model.

1. **Transformation _(The Polish)_**:
This head takes the final vector $T_i$ to a small dense layer with an activation function _(GELU)_ and Layer Normalization. This acts like a final "processing" step to prepare the vector for decoding.

$$
x = LayerNorm(GELU(T_i \cdot W_{transform} + b_{transform}))
$$

2. **Projection _(The Decoder)_**:
Now we face a big problem: We have a vector of size **768**, but. we need scores forr **30,522** tokens. We need a matrix that connects the _Hidden Size_ to _Vocabulary Size_.

**The Trick _(Weight Tying)_**: Instead of learning a new giant matrix, BERT reuses the **Input Token Embedding Matrix** _($E_{tok}$)_ we started with!

We multiply our vector $x$ by the **transpose** of the embedding matrix:

$$
Logits = x \cdot E_{tok}^T + bias
$$ 

Logits mean: **Scores** for every token in the vocabulary.

3. **Softmax**:
We take those 30,522 "logits" _(row scores)_ and squish them into probabilities that sum to 1.

$$
P = \text{Softmax}(Logits)
$$

Now we have a probability distribution over the vocabulary for every token. The loss is:

$$
L_{MLM} = -\log (P("bank"))
$$

#### NSP Loss

We also need to check if Sentence B actually followed Sentence A. For this, we look **only** at the `[CLS]` token's final vector $T_{CLS}$.

1. **Projection**:
We multiply $T_{CLS}$ by a small weight matrix $W_{NSP}$ _(shape 768 x 2)_ to get two scores: "IsNext" and "NotNext".
2. **Probability**:
We apply **Softmax**.
3. **Error**:
We check the probability of the true label.

$$
L_{NSP} = -\log (P(TrueLabel))
$$

---

The total loss is the sum of the two losses:

$$
L_{total} = L_{MLM} + L_{NSP}
$$

