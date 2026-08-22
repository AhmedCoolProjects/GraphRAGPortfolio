# LSTM

We previously saw how standard RNNs can suffer from the **vanishing gradient problem**, which makes it difficult for them to learn long-term dependencies in sequences. To address this issue, a more advanced type of RNN called **Long Short-Term Memory (LSTM)** was developed.

The big breakthrough is that LSTMs split the **memory** into two separate parts:
1. **Hidden State $h_t$**: This is similar to short-term memory in standard RNNs. It captures information about the current input and recent past and used for immediate predictions.
2. **Cell State $C_t$**: This acts as long-term memory. It is like a *superhighway* that runs through the entire chain with very few interactions. Information can flow along it for a long time without being changed or *faded out*.

The LSTM uses structures called **gates** to carefully control what gets added to or removed from the cell state.

## Cell State

The cell state $C_t$ carries long-term information from the past $C_{t-1}$ **straight** to the present $C_t$ with minimal changes.

In standard RNNs, the hidden state $h_t$ is updated at every time step and constantly being squashed by **tanh** and transformed by weight matrices, which can lead to loss of information over time.

In an LSTM, the cell state flows mostly **linearly**. We can add things to it or remove things from it, but if we do nothing, the information just coasts along unchanged. This helps preserve long-term dependencies.

Since $C_t$ is mostly linear, the error signal _(gradient)_ can flow backward along it without being constantly multiplied by small derivatives from activation functions. This means the error doesn't "vanish", and the network can learn from data that happened many time steps ago.

Now we have this $C_t$, we need a way to control what information gets added or removed from it. For instance, *forget the subject is "singular" when we see a period*, or *remember the new subject is "plural" when we see "and"*.

These mechanisms are called **Gates**.
- **Forget Gate**: Drop old, irrelevant info.
- **Input Gate**: Add new, relevant info.
- **Output Gate**: Decide what to output based on cell state.

## Gates

The LSTM has three main gates, all of them are neural networks with a **sigmoid activation** function. All these gates contribute to updating the cell state $C_t$ as follows:

$$
\boxed{C_t = f_t * C_{t-1} + i_t * \tilde{C}_t}
$$

Where:
- $f_t$ is the **forget gate** vector, which decides what information to discard from the previous cell state $C_{t-1}$.
- $i_t$ is the the **input gate** vector, which decides what new information to add to the cell state.
- $\tilde{C}_t$ is the candidate values vector, which contains the new information that could be added to the cell state.

### Forget Gate

The forget gate is the first step in the LSTM cell. It decides what information we should throw away or "forget" from the cell state.

The forget gate looks at:
1. **$h_{t-1}$**: The previous hidden state (short-term memory).
2. **$x_t$**: The current input (the new data).

It squashes them together and passes them through a **sigmoid function $\sigma$**.

$$
\boxed{f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)}
$$

Since the sigmoid outputs values between 0 and 1, each number in $f_t$ represents how much of each component in the cell state $C_{t-1}$ we should keep. A value of 1 means "completely keep this" while a value of 0 means "completely get rid of this".

### Input Gate

This gate decides what new information to store in the Cell State. It has two parts that work together:
1. **Candidate Layer $\tilde{C}_t$**: This creates a vector of new candidate values that could be added to the state. It uses a **tanh** activation to create values between -1 and 1.
   - Why tanh? It allows us to add (positive) or subtract (negative) information from the cell state.
2. **Input Gate Layer $i_t$**: This is another sigmoid layer _(outputing 0 to 1)_. It acts as a filter for the candidate values.

The equations are:

$$
\boxed{\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)}
$$

$$
\boxed{i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)}
$$

### Output Gate

Decides what we should output actually right now as the **Hidden State** $h_t$. The cell state $C_t$ might contain a lot of information that is useful for the future but not relevant for this exact moment.

For example, the cell might know "The subject is Bob (singular)", which is important for a verb coming up later. But if the current word is "is", we don't need to output that information right now since "is" carries that information itself.

The Output Gate handles this:
1. **Filter $o_t$**: It uses a sigmoid layer to decide which parts of the cell state are relevant to output now.
2. **Squash**: It takes the Cell State $C_t$ and puts it through **tanh** _(to get values between -1 and 1)_.
3. **Calculate**: It multiplies the squashed cell state by the filter to get the final hidden state $h_t$.

$$
\boxed{o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)}
$$

$$
\boxed{h_t = o_t * \tanh(C_t)}
$$

:::info Analogy
Think of the Cell State as a **backpack** and the Hidden State as **your hands**. 
- The **Cell State** (backpack) carries everything you might need for the long journey ahead. It holds long-term information _(like "Bob is the subject", "The story is set in the past", "We are talking about food")_ safely inside, protecting it from getting lost or forgotten.
- The **Hidden State** (your hands) only hold what is useful **right now** for the immediate task at hand.

:::

## Forward Pass

Because we have 3 gates _(Forget, Input, Output)_ plus the _Candidate Layer_, we essentially have 4 separate _mini-networks_ inside the LSTM cell.

So we need 4 sets of these weights and biases:
- Forget Gate: $W_f$, $b_f$
- Input Gate: $W_i$, $b_i$
- Candidate Layer: $W_C$, $b_C$
- Output Gate: $W_o$, $b_o$

This why LSTMs have **4x more parameters** than standard RNNs of the same hidden size.

Before coding the Forward Pass, let's summarize the steps and equations involved in an LSTM cell at time step $t$:

1. **Forget Gate**: Decide what to forget from previous cell state.
   $$
   f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
   $$

2. **Input Gate**: Decide what new information to add.
   $$
   i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
   $$
3. **Candidate Layer**: Create new candidate values.
   $$
   \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
   $$
4. **Update Cell State**: Combine old cell state and new candidate values.
   $$
   C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
   $$
5. **Output Gate**: Decide what to output.
   $$
   o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
   $$
6. **Compute Hidden State**: Final output for this time step.
   $$
   h_t = o_t * \tanh(C_t)
   $$
7. **Compute Output at that Step**: 
   $$
   y_t = \text{softmax}(W_{hy} \cdot h_t + b_y)
   $$

## BPTT

The Backpropagation Through Time (BPTT) for LSTMs is more complex than standard RNNs due to the additional gates and the cell state. However, the key idea remains the same: we need to compute gradients for all parameters by applying the chain rule through time.

Before diving into the code, it's important to note that during backpropagation, we need to compute gradients for:
- Weights and biases for all gates: $W_f$, $b_f$, $W_i$, $b_i$, $W_c$, $b_c$, $W_o$, $b_o$
- The cell state $C_t$ and hidden state $h_t$ at each time step.

Below is the math behind the BPTT for LSTMs:
1. Compute gradients of the loss with respect to the output $h_t$ and cell state $C_t$.
2. Backpropagate through the output gate to get gradients for $o_t$ since it affects $h_t$.
3. Backpropagate through the cell state update to get gradients for $f_t$, $i_t$, and $\tilde{C}_t$.
4. Backpropagate through the forget and input gates to get gradients for their weights and biases.
5. Accumulate gradients over all time steps.

The used loss function here is the same cross-entropy loss as in standard RNNs.

Let's compute the gradients step-by-step:
1. **Gradients for Output Layer**:
    $$
    d\hat{y}_t = \hat{y}_t - y_t = \delta
    $$
    $$
    dW_{hy} += \delta \cdot h_t^T
    $$
    $$
    db_y += \delta
    $$
2. **Gradients for Hidden State**:
    $$
    dh_t = W_{hy}^T \cdot \delta + dh_{next}
    $$
3. **Gradients for Output Gate**: Let's denote $[h_{t-1}, x_t]$ as $concat_t$ for simplicity.
    $$
    do_t = dh_t * \tanh(C_t) * o_t * (1 - o_t)
    $$
    $$
    dW_o += do_t \cdot concat_t^T
    $$
    $$
    db_o += do_t
    $$
4. **Gradients for Cell State**:
    $$
    dC_t = dh_t * o_t * (1 - \tanh^2(C_t)) + dC_{next}
    $$
5. **Gradients for Forget Gate**:
    $$
    df_t = dC_t * C_{t-1} * f_t * (1 - f_t)
    $$
    $$
    dW_f += df_t \cdot concat_t^T
    $$
    $$
    db_f += df_t
    $$
6. **Gradients for Input Gate**:
    $$
    di_t = dC_t * \tilde{C}_t * i_t * (1 - i_t)
    $$
    $$
    dW_i += di_t \cdot concat_t^T
    $$
    $$
    db_i += di_t
    $$
7. **Gradients for Candidate Layer**:
    $$
    d\tilde{C}_t = dC_t * i_t * (1 - \tilde{C}_t^2)
    $$
    $$
    dW_C += d\tilde{C}_t \cdot concat_t^T
    $$
    $$
    db_C += d\tilde{C}_t
    $$




## Python Implementation

```python title="lstm.py"
import numpy as np

class LSTM:
    def __init__(self, hidden_size, vocab_size):
        self.H = hidden_size
        self.V = vocab_size

        # Initialize weights (4 sets)
        # f: Forget, i: Input, c: Candidate, o: Output
        self._W_size = (self.H, self.H + self.V)  # weights shape for all gates
        # Forget Gate
        self.Wf = np.random.randn(*self._W_size) * 0.01
        self.bf = np.zeros((self.H, 1))

        # Input Gate
        self.Wi = np.random.randn(*self._W_size) * 0.01
        self.bi = np.zeros((self.H, 1))

        # Candidate Layer
        self.Wc = np.random.randn(*self._W_size) * 0.01
        self.bc = np.zeros((self.H, 1))

        # Output Gate
        self.Wo = np.random.randn(*self._W_size) * 0.01
        self.bo = np.zeros((self.H, 1))

        # Final Output Layer
        self.Why = np.random.randn(self.V, self.H) * 0.01
        self.by = np.zeros((self.V, 1))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _tanh(self, z):
        return np.tanh(z)

    def _softmax(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z, axis=0)

    def _sigmoid_derivative(self, z):
        return z * (1 - z)

    def _tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def forward(self, inputs, h_prev, c_prev):
        '''
        inputs: list of input indices
        h_prev: previous hidden state (H x 1)
        c_prev: previous cell state (H x 1)
        '''

        xs, hs, cs, zs, ys = {}, {}, {}, {}, {} # store values for each time step
        fs, is_, cs_tilde, os = {}, {}, {}, {} # gate activations
        concat_inputs = {}

        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)

        total_cost = 0

        for t in range(len(inputs)):
            # 1. One-hot encode the input character
            xs[t] = np.zeros((self.V, 1))
            xs[t][inputs[t]] = 1

            # 2. Concatenate h_prev and x_t
            concat_inputs[t] = np.vstack((hs[t-1], xs[t]))  # (H + V) x 1

            # 3. Forget Gate
            fs[t] = self._sigmoid(np.dot(self.Wf, concat_inputs[t]) + self.bf)

            # 4. Input Gate
            is_[t] = self._sigmoid(np.dot(self.Wi, concat_inputs[t]) + self.bi)

            # 5. Candidate Layer
            cs_tilde[t] = self._tanh(np.dot(self.Wc, concat_inputs[t]) + self.bc)

            # 6. Output Gate
            os[t] = self._sigmoid(np.dot(self.Wo, concat_inputs[t]) + self.bo)

            # 7. Update Cell State
            cs[t] = fs[t] * cs[t-1] + is_[t] * cs_tilde[t] # (H x 1)

            # 8. Compute Hidden State
            hs[t] = os[t] * self._tanh(cs[t])  # (H x 1)

            # 9. Compute Output
            zs[t] = np.dot(self.Why, hs[t]) + self.by
            ys[t] = self._softmax(zs[t])  # (V x 1)

        cache = (xs, hs, cs, fs, is_, cs_tilde, os, zs, ys, concat_inputs)

        return ys, hs[len(inputs)-1], cs[len(inputs)-1], cache
    
    def compute_cost(self, y_preds, targets):
        total_cost = 0
        for t in range(len(targets)):
            prob_of_target = y_preds[t][targets[t], 0] # [target_index, 0] gives the prob of that target and the 0 is to get the scalar from the (1,1) array
            total_cost += -np.log(prob_of_target + 1e-9) # add small value to avoid log(0)
        return total_cost / len(targets)

    def backpropagation(self, targets, cache):
        xs, hs, cs, fs, is_, cs_tilde, os, zs, ys, concat_inputs = cache
        # Initialize gradients
        self.dWf = np.zeros_like(self.Wf)
        self.dbf = np.zeros_like(self.bf)
        self.dWi = np.zeros_like(self.Wi)
        self.dbi = np.zeros_like(self.bi)
        self.dWc = np.zeros_like(self.Wc)
        self.dbc = np.zeros_like(self.bc)
        self.dWo = np.zeros_like(self.Wo)
        self.dbo = np.zeros_like(self.bo)
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(hs[0])
        dc_next = np.zeros_like(cs[0])

        for t in reversed(range(len(targets))):
            # 1. Output layer
            dy = np.copy(ys[t])
            dy[targets[t]] -= 1 # y_pred - y_true
            self.dWhy += np.dot(dy, hs[t].T)
            self.dby += dy

            # 2. Gradient for hidden state dh_t
            dh = np.dot(self.Why.T, dy) + dh_next

            # 3. Gradient for output gate
            do = dh * self._tanh(cs[t]) * os[t] * (1 - os[t])
            self.dWo += np.dot(do, concat_inputs[t].T)
            self.dbo += do

            # 4. Gradient for cell state
            dc = dh * os[t] * (1 - self._tanh(cs[t])**2) + dc_next

            # 5. Gradient for forget gate
            df = dc * cs[t-1] * fs[t] * (1 - fs[t])
            self.dWf += np.dot(df, concat_inputs[t].T)
            self.dbf += df

            # 6. Gradient for input gate
            di = dc * cs_tilde[t] * is_[t] * (1 - is_[t])
            self.dWi += np.dot(di, concat_inputs[t].T)
            self.dbi += di

            # 7. Gradient for candidate layer
            dc_tilde = dc * is_[t] * (1 - cs_tilde[t]**2)
            self.dWc += np.dot(dc_tilde, concat_inputs[t].T)
            self.dbc += dc_tilde

            # 8. Gradient for concatenated input
            dconcat = (np.dot(self.Wf.T, df) +
                        np.dot(self.Wi.T, di) +
                        np.dot(self.Wc.T, dc_tilde) +
                        np.dot(self.Wo.T, do))
            dh_next = dconcat[:self.H, :]  # Gradient for h_(t-1)
            dc_next = dc * fs[t]          # Gradient for C_(t-1)

        # Gradient clipping to prevent exploding gradients
        for grad in [self.dWf, self.dbf, self.dWi, self.dbi,
                     self.dWc, self.dbc, self.dWo, self.dbo,
                     self.dWhy, self.dby]:
            np.clip(grad, -5, 5, out=grad)

    def update_parameters(self, learning_rate=0.01):
        self.Wf -= learning_rate * self.dWf
        self.bf -= learning_rate * self.dbf
        self.Wi -= learning_rate * self.dWi
        self.bi -= learning_rate * self.dbi
        self.Wc -= learning_rate * self.dWc
        self.bc -= learning_rate * self.dbc
        self.Wo -= learning_rate * self.dWo
        self.bo -= learning_rate * self.dbo
        self.Why -= learning_rate * self.dWhy
        self.by -= learning_rate * self.dby

    def sample(self, seed_idx, h_prev, c_prev, length=20):
        x = np.zeros((self.V, 1))
        x[seed_idx] = 1
        indices = []

        for t in range(length):
            concat_input = np.vstack((h_prev, x))

            # Gates
            f = self._sigmoid(np.dot(self.Wf, concat_input) + self.bf)
            i = self._sigmoid(np.dot(self.Wi, concat_input) + self.bi)
            c_tilde = self._tanh(np.dot(self.Wc, concat_input) + self.bc)
            o = self._sigmoid(np.dot(self.Wo, concat_input) + self.bo)

            # Update cell state
            c = f * c_prev + i * c_tilde

            # Compute hidden state
            h = o * self._tanh(c)

            z = np.dot(self.Why, h) + self.by
            y = self._softmax(z)

            # Sample from the probability distribution
            idx = np.random.choice(range(self.V), p=y.ravel()) # this gives us a scalar index meaning the index of the predicted character. the .ravel() is to convert the (V,1) shape to (V,) shape which is required by np.random.choice, this .choice samples according to the probabilities in y
            x = np.zeros((self.V, 1))
            x[idx] = 1
            indices.append(idx)
            h_prev = h
            c_prev = c

        return indices
```