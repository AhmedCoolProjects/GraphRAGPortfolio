# Bi-LSTM

**Bi-directional Long Short-Term Memory (Bi-LSTM)** is a powerful variant of the traditional LSTM architecture that processes data in both *forward* and *backward* directions.

The core idea solves a specific limitation: standard LSTMs only look at the **past** context. But in many tasks, the future context is just as important.

:::note Situation

Think about fill-in-the-blank:
> "He said, ___." _(Could be anything: hello, goodbye, stop, go, etc.)_

> "He said, ___ are you?" _(Now we know it's probably "how" or "who")_

The word "are" _(which comes after the blank)_ gave us the crucial clue. A standard LSTM reads left-to-right and wouldn't see "are" until it's too late.
:::

## Architecture

A Bi-LSTM isn't a new type of cell; it's a **new architecture** using two existing LSTM layers.
1. **Forward Layer**: Reads the sequence from start to end ($x_1 \rightarrow x_T$).
2. **Backward Layer**: Reads the sequence from end to start ($x_T \rightarrow x_1$).
3. **Combination**: At every time step $t$, we combine the hidden states from both layers _(usually by concatenation)_.
   - If the forward LSTM produces a hidden state $h_t^{(f)}$ and the backward LSTM produces $h_t^{(b)}$, the combined output is:
   $$
   h_t = [h_t^{(f)}; h_t^{(b)}]
   $$

This gives the network a complete view of the entire sequence, past and future, at every single point in time.

## Implementation

We don't need to write complex new math equations. We can simply **reuse** the **LSTM** cell we already built.

A Bi-LSTM is just a "wrapper" class that manages two separate LSTM instances. Here's is a description of the steps involved:
1. **Initialize Two LSTMs**: One for the forward pass and one for the backward pass.
2. **Bi-LSTM Forward Method**:
    - Pass the input sequence normally to the Forward LSTM
    - **Reverse** the input sequence and pass it to the Backward LSTM
    - **Reverse** the output from the Backward LSTM again _(to get it back in the original order: $t=1$ to $t=T$)_
    - **Concatenate** the outputs from both LSTMs at each time step
3. **Return** the combined output sequence

## Python Implementation

```python
import numpy as np

class LSTM:
    # ...
    pass

class BiLSTM:
    def __init__(self, hidden_size, vocab_size, learning_rate=0.01):
        self.H = hidden_size
        self.V = vocab_size
        self.lr = learning_rate

        # Two LSTM instances
        self.f_lstm = LSTM(self.H, self.V, self.lr)  # Forward LSTM
        self.b_lstm = LSTM(self.H, self.V, self.lr)  # Backward L

        # Combined output layer
        # The input to this layer is (H_fwd + H_bwd) = 2H
        self.Wy = np.random.randn(self.V, 2 * self.H) * 0.01
        self.by = np.zeros((self.V, 1))

        # Gradients for output layer
        self.dWy = np.zeros_like(self.Wy)
        self.dby = np.zeros_like(self.by)
    
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0))
        return exp_z / exp_z.sum(axis=0)

    def forward(self, inputs, h_prev_f, c_prev_f, h_prev_b, c_prev_b):
        # 1. Run forward LSTM
        _, _, _, cache_f = self.f_lstm.forward(inputs, h_prev_f, c_prev_f)
        _, hs_f, _, _, _, _, _, _, _, _ = cache_f

        # 2. Run backward LSTM
        rev_inputs = inputs[::-1]
        _, _, _, cache_b = self.b_lstm.forward(rev_inputs, h_prev_b, c_prev_b)
        _, rev_hs_b, _, _, _, _, _, _, _, _ = cache_b

        # 3. Combine outputs
        ys, concat_hs = {}, {}

        # We need to match time steps.
        # hs_f[0] corresponds to inputs[0]
        # rev_hs_b[0] corresponds to inputs[T-1] (last time step)

        T = len(inputs)
        for t in range(T):
            h_f = hs_f[t] # forward hidden state for input t
            h_b = rev_hs_b[T - 1 - t] # backward hidden state for input t

            concat_hs[t] = np.vstack((h_f, h_b))  # Concatenate
            z = np.dot(self.Wy, concat_hs[t]) + self.by
            ys[t] = self._softmax(z)
    
        cache = (cache_f, cache_b, concat_hs, ys)
        return ys, cache

    def compute_cost(self, ys, targets):
        total_cost = 0
        for t in range(len(targets)):
            prob = ys[t][targets[t], 0]
            total_cost += -np.log(prob + 1e-9)
        return total_cost / len(targets)

    def backpropagation(self, targets, cache):
        cache_f, cache_b, concat_hs, ys = cache

        T = len(targets)

        # We need to collect gradients for the hidden states of both LSTMs
        # to run their specific backprop logic.
        dh_f_seq = {}
        dh_b_seq = {}

        # 1. Backprop through the Combined output layer
        for t in reversed(range(T)):
            dy = np.copy(ys[t])
            dy[targets[t]] -= 1

            self.dWy += np.dot(dy, concat_hs[t].T)
            self.dby += dy

            # Calculate gradient for the concatenated hidden state
            # dh_concat = Wy.T @ dy
            dh_concat = np.dot(self.Wy.T, dy)

            # Split gradients back to forward and backward LSTM
            dh_f_seq[t] = dh_concat[:self.H, :]
            dh_b_seq[T - 1 - t] = dh_concat[self.H:, :]

        # 2. Backprop through Forward LSTM
        self._lstm_backprop_manual(self.f_lstm, dh_f_seq, cache_f, T)
        # 3. Backprop through Backward LSTM
        self._lstm_backprop_manual(self.b_lstm, dh_b_seq, cache_b, T)

    def _lstm_backprop_manual(self, lstm, dh_seq, cache, T):
        '''
        Runs BPTT for a single LSTM given a sequence of hidden states gradients.
        '''
        xs, hs, cs, _, _, fs, is_, cs_cand, os, concat_inputs = cache
        
        # Initialize LSTM gradients
        lstm.dWf = np.zeros_like(lstm.Wf)
        lstm.dWi = np.zeros_like(lstm.Wi)
        lstm.dWc = np.zeros_like(lstm.Wc)
        lstm.dWo = np.zeros_like(lstm.Wo)
        lstm.dbf = np.zeros_like(lstm.bf)
        lstm.dbi = np.zeros_like(lstm.bi)
        lstm.dbc = np.zeros_like(lstm.bc)
        lstm.dbo = np.zeros_like(lstm.bo)

        dh_next = np.zeros_like(hs[0])
        dc_next = np.zeros_like(cs[0])

        for t in reversed(range(T)):
            # Get the gradient from the combined layer above
            dh_from_above = dh_seq[t]
            # Total gradient for h_t
            dh = dh_from_above + dh_next

            # ---- Standard BPTT ----
            do = dh * lstm._tanh(cs[t])
            do_raw = do * lstm._sigmoid_derivative(os[t])
            lstm.dWo += np.dot(do_raw, concat_inputs[t].T)
            lstm.dbo += do_raw

            dc = dh * os[t] * lstm._tanh_derivative(cs[t]) + dc_next

            dc_cand = dc * is_[t]
            dc_cand_raw = dc_cand * lstm._tanh_derivative(cs_cand[t])
            lstm.dWc += np.dot(dc_cand_raw, concat_inputs[t].T)
            lstm.dbc += dc_cand_raw

            di = dc * cs_cand[t]
            di_raw = di * lstm._sigmoid_derivative(is_[t])
            lstm.dWi += np.dot(di_raw, concat_inputs[t].T)
            lstm.dbi += di_raw

            df = dc * cs[t-1]
            df_raw = df * lstm._sigmoid_derivative(fs[t])
            lstm.dWf += np.dot(df_raw, concat_inputs[t].T)
            lstm.dbf += df_raw

            dconcat = np.dot(lstm.Wf.T, df_raw)
            dconcat += np.dot(lstm.Wi.T, di_raw)
            dconcat += np.dot(lstm.Wc.T, dc_cand_raw)
            dconcat += np.dot(lstm.Wo.T, do_raw)

            dh_next = dconcat[:lstm.H, :]
            dc_next = dc * fs[t]

        # Clip gradients
        for grad in [lstm.dWf, lstm.dWi, lstm.dWc, lstm.dWo, lstm.dbf, lstm.dbi, lstm.dbc, lstm.dbo]:
            np.clip(grad, -5, 5, out=grad)
    
    def update_parameters(self):
        # Combined Layer
        self.Wy -= self.lr * self.dWy
        self.by -= self.lr * self.dby
        
        # Inner LSTMs
        self._lstm_update_manual(self.f_lstm)
        self._lstm_update_manual(self.b_lstm)
    
    def _lstm_update_manual(self, lstm):
        lstm.Wf -= self.lr * lstm.dWf
        lstm.Wi -= self.lr * lstm.dWi
        lstm.Wc -= self.lr * lstm.dWc
        lstm.Wo -= self.lr * lstm.dWo
        lstm.bf -= self.lr * lstm.dbf
        lstm.bi -= self.lr * lstm.dbi
        lstm.bc -= self.lr * lstm.dbc
        lstm.bo -= self.lr * lstm.dbo

    def sample(self, seed_idx, h_prev_fwd, c_prev_fwd, length=20):
        """
        Generates text using ONLY the Forward LSTM.
        Ideally, Bi-LSTMs are not used for generation, but this is a demonstration.
        We feed zeros for the backward state.
        """
        x = np.zeros((self.V, 1))
        x[seed_idx] = 1
        
        generated_indices = []
        
        # Placeholder for backward state (zeros)
        h_fake_bwd = np.zeros((self.H, 1))
        
        for t in range(length):
            # --- 1. Run One Step of Forward LSTM ---
            # (We manually run the step logic here since lstm.forward_pass does a loop)
            
            # Concatenate input [h_prev, x]
            concat_input = np.vstack((h_prev_fwd, x))
            
            # Gates
            f = self.f_lstm._sigmoid(np.dot(self.f_lstm.Wf, concat_input) + self.f_lstm.bf)
            i = self.f_lstm._sigmoid(np.dot(self.f_lstm.Wi, concat_input) + self.f_lstm.bi)
            c_cand = self.f_lstm._tanh(np.dot(self.f_lstm.Wc, concat_input) + self.f_lstm.bc)
            o = self.f_lstm._sigmoid(np.dot(self.f_lstm.Wo, concat_input) + self.f_lstm.bo)
            
            # Update states
            c = f * c_prev_fwd + i * c_cand
            h = o * self.f_lstm._tanh(c)
            
            # --- 2. Combine with Fake Backward State ---
            concat_hs = np.vstack((h, h_fake_bwd))
            
            # --- 3. Output ---
            z = np.dot(self.Wy, concat_hs) + self.by
            y_pred = self._softmax(z)
            
            # Sample
            idx = np.random.choice(range(self.V), p=y_pred.ravel())
            
            x = np.zeros((self.V, 1))
            x[idx] = 1
            generated_indices.append(idx)
            
            # Update Forward states
            h_prev_fwd = h
            c_prev_fwd = c
            
        return generated_indices
```