# GRU

**Gated Recurrent Unit _(GRU)_** invented in 2014 _(much later than LSTM)_ to solve the same vanishing gradient problem but with a simpler design.

The main differences between GRU and LSTM are:
1. **Two Gates instead of Three**: The GRU combines the forget and input gates into a single **Update Gate**. It also has a **Reset Gate**.
2. **One State instead of Two**: GRU merges the Cell State ($C_t$) and Hidden State ($h_t$) into a single hidden state ($h_t$).

This means the GRU has **fewer parameters** than LSTM, making it faster to train and less prone to overfitting on smaller datasets, while often performing just as well.

## Gates in GRU

In GRU, we calculate everything using the previous hidden state ($h_{t-1}$) and the current input ($x_t$).

The two gates look very similar to the LSTM gates, using the sigmoid ($\sigma$) activation function to produce outputs between 0 and 1.

### Reset Gate

The **Reset Gate** ($r_t$) decides how much of the past information to ignore _(reset)_ when calculating the new candidate memory. If $r_t = 0$, the network completely forgets the previous hidden state for the candidate memory being treated as the start of the memory.

$$
\boxed{r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)}
$$

Where:
- $W_r$ is the weight matrix for the reset gate.
- $b_r$ is the bias for the reset gate.

### Update Gate

The **Update Gate** ($z_t$) is the multitasker. It acts as both the "forget" and "input" gate combined. It decides: "How much of the old memory should I keep, versus how much of the new candidate memory should I add?"

$$
\boxed{z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)}
$$

Where:
- $W_z$ is the weight matrix for the update gate.
- $b_z$ is the bias for the update gate.
  
## Candidate Memory

This is where the new memory is proposed. It uses $\tanh$ _(like the LSTM candidate)_, but with the **reset gate** applied to the previous hidden state ($h_{t-1}$). This allows the model to forget parts of the past when creating the new candidate memory.

$$
\boxed{\tilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1}, x_t] + b_h)}
$$

Where:
- $W_h$ is the weight matrix for the candidate memory.
- $b_h$ is the bias for the candidate memory.
- $*$ denotes element-wise multiplication.

## Final Memory (Hidden State)

This is the elegant part. Instead of separated addition and deletion steps, the GRU uses the **Update Gate** ($z_t$) to **slide** between the old memory and the new candidate.

$$
\boxed{h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t}
$$

Where:
- $h_t$ is the new hidden state (final memory) at time step $t.
- $h_{t-1}$ is the previous hidden state.
- $\tilde{h}_t$ is the candidate memory.
- $z_t$ is the update gate.
- $*$ denotes element-wise multiplication.
  
## BPTT for GRU

The Backpropagation Through Time (BPTT) for GRU follows the same principles as for LSTM and vanilla RNNs. The gradients are computed with respect to the loss function by unrolling the GRU through time and applying the chain rule.

The main difference lies in the specific equations for the gates and candidate memory, which affect how gradients are calculated. However, the overall process of accumulating gradients over time steps and updating weights remains consistent with other RNN architectures.

Let's start by computing the gradients of the loss ($L$) with respect to the output $\hat{y}_t$:

$$
\frac{\partial L}{\partial \hat{y}_t} = \hat{y}_t - y_{target} = d_y
$$

_(This is the standard gradient for Softmax + Cross-Entropy loss)_

Then using the Chain Rule:

$$
\boxed{dW_y += d_y \cdot h_t^T} \\
\boxed{db_y += d_y}
$$

Next, we compute the gradient with respect to the hidden state $h_t$:

$$
dh = W_y^T \cdot d_y + dh_{next}
$$

Then,

$$
dz = dh * (\tilde{h}_t - h_{t-1}) * \sigma'(z_t) \\
d\tilde{h} = dh * z_t * \tanh'(\tilde{h}_t) \\
dr = (W_h[:, :H]^T \cdot d\tilde{h}) * h_{t-1} * \sigma'(r_t)
$$

Then we compute the gradients for the weights and biases:

$$
\boxed{dW_z += dz \cdot [h_{t-1}, x_t]^T} \\
\boxed{db_z += dz} \\
\boxed{dW_r += dr \cdot [h_{t-1}, x_t]^T} \\
\boxed{db_r += dr} \\
\boxed{dW_h += d\tilde{h} \cdot [r_t * h_{t-1}, x_t]^T} \\
\boxed{db_h += d\tilde{h}}
$$

Finally, we compute the gradient w.r.t the previous hidden state $h_{t-1}$:

$$
d_{from\_h} = dh * (1 - z_t) \\
d_{from\_cand} = (W_h[:, :H]^T \cdot d\tilde{h}) * r_t \\
d_{from\_update} = W_z[:, :H]^T \cdot dz \\
d_{from\_reset} = W_r[:, :H]^T \cdot dr
$$

$$
\boxed{dh_{next} = d_{from\_h} + d_{from\_cand} + d_{from\_update} + d_{from\_reset}}
$$



## Python Implementation

Here is a simple implementation of a GRU cell in Python using NumPy:

```python
import numpy as np

class GRU:
    def __init__(self, hidden_size, vocab_size, learning_rate=0.01):
        self.H = hidden_size
        self.V = vocab_size
        self.lr = learning_rate

        # Weights Initialization
        W_shape = (self.H, self.H + self.V)

        self.Wr = np.random.randn(*W_shape) * 0.01
        self.br = np.zeros((self.H, 1))

        self.Wz = np.random.randn(*W_shape) * 0.01
        self.bz = np.zeros((self.H, 1))

        self.Wh = np.random.randn(*W_shape) * 0.01
        self.bh = np.zeros((self.H, 1))

        self.Wy = np.random.randn(self.V, self.H) * 0.01
        self.by = np.zeros((self.V, 1))

        # Gradients
        self.dWr, self.dWz, self.dWh, self.dWy = [None]*4
        self.dbr, self.dbz, self.dbh, self.dby = [None]*4

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_derivative(self, a):
        return 1 - a ** 2
    
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum(axis=0)

    def forward(self, inputs, h_prev):
        xs, hs, zs, ys = {}, {}, {}, {}
        rs_g, zs_g, h_cands = {}, {}, {}
        concat_inputs = {}

        hs[-1] = np.copy(h_prev)

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.V, 1))
            xs[t][inputs[t]] = 1

            concat_inputs[t] = np.vstack((hs[t-1], xs[t]))

            rs_g[t] = self._sigmoid(np.dot(self.Wr, concat_inputs[t]) + self.br)
            zs_g[t] = self._sigmoid(np.dot(self.Wz, concat_inputs[t]) + self.bz)
            h_cands = self._tanh(np.dot(self.Wh, np.vstack((rs_g[t] * hs[t-1], xs[t]))) + self.bh)

            hs[t] = (1 - zs_g[t]) * hs[t-1] + zs_g[t] * h_cands

            # Final Prediction
            zs[t] = np.dot(self.Wy, hs[t]) + self.by
            ys[t] = self._softmax(zs[t])

        cache = (xs, hs, zs, ys, rs_g, zs_g, h_cands, concat_inputs)
        return ys, hs[len(inputs)-1], cache

    def compute_cost(self, ys, targets):
        total_cost = 0
        for t in range(len(targets)):
            prob = ys[t][targets[t], 0]
            total_cost += -np.log(prob + 1e-9)
        return total_cost / len(targets)

    def backpropagation(self, targets, cache):
        xs, hs, zs, ys, rs_g, zs_g, h_cands, concat_inputs = cache
        T = len(targets)

        self.dWr = np.zeros_like(self.Wr)
        self.dWz = np.zeros_like(self.Wz)
        self.dWh = np.zeros_like(self.Wh)
        self.dWy = np.zeros_like(self.Wy)
        self.dbr = np.zeros_like(self.br)
        self.dbz = np.zeros_like(self.bz)
        self.dbh = np.zeros_like(self.bh)
        self.dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(T)):
            dy = np.copy(ys[t])
            dy[targets[t]] -= 1

            self.dWy += np.dot(dy, hs[t].T)
            self.dby += dy

            dh = np.dot(self.Wy.T, dy) + dh_next

            dz = dh * (h_cands[t] - hs[t-1]) * self._sigmoid_derivative(zs_g[t])
            self.dWz += np.dot(dz, concat_inputs[t].T)
            self.dbz += dz

            dh_cand = dh * zs_g[t] * self._tanh_derivative(h_cands[t])
            self.dWh += np.dot(dh_cand, np.vstack((rs_g[t] * hs[t-1], xs[t])).T)
            self.dbh += dh_cand

            dr = np.dot(self.Wh[:, :self.H].T, dh_cand) * hs[t-1] * self._sigmoid_derivative(rs_g[t])
            self.dWr += np.dot(dr, concat_inputs[t].T)
            self.dbr += dr

            # Gradient for Previous Hidden State (dh_next) at time t-1
            # a. From Linear Combination: dh_next += dh * (1 - zs_g[t])
            d_from_linear = dh * (1 - zs_g[t])
            # b. From Candidate Memory: dh_next += (W_h[:, :H].T * dh_cand) * rs_g[t]
            d_from_cand = np.dot(self.Wh[:, :self.H].T, dh_cand) * rs_g[t]
            # c. From Update Gate: dh_next += np.dot(W_z[:, :H].T * dz)
            d_from_update = np.dot(self.Wz[:, :self.H].T, dz)
            # d. From Reset Gate: dh_next += np.dot(W_r[:, :H].T * dr)
            d_from_reset = np.dot(self.Wr[:, :self.H].T, dr)

            dh_next = d_from_linear + d_from_cand + d_from_update + d_from_reset

        # Gradient Clipping
        for grad in [self.dWr, self.dWz, self.dWh, self.dWy, self.dbr, self.dbz, self.dbh, self.dby]:
            np.clip(grad, -5, 5, out=grad)
    
    def update_parameters(self):
        self.Wr -= self.lr * self.dWr
        self.Wz -= self.lr * self.dWz
        self.Wh -= self.lr * self.dWh
        self.Wy -= self.lr * self.dWy
        self.br -= self.lr * self.dbr
        self.bz -= self.lr * self.dbz
        self.bh -= self.lr * self.dbh
        self.by -= self.lr * self.dby
    
    def sample(self, seed_idx, h_prev, length=20):
        x = np.zeros((self.V, 1))
        x[seed_idx] = 1
        indices = []

        for t in range(length):
            concat_input = np.vstack((h_prev, x))

            r = self._sigmoid(np.dot(self.Wr, concat_input) + self.br)
            z = self._sigmoid(np.dot(self.Wz, concat_input) + self.bz)
            h_cand = self._tanh(np.dot(self.Wh, np.vstack((r * h_prev, x))) + self.bh)
            h = (1 - z) * h_prev + z * h_cand
            y = self._softmax(np.dot(self.Wy, h) + self.by)
            idx = np.random.choice(range(self.V), p=y.ravel())
            indices.append(idx)
            x = np.zeros((self.V, 1))
            x[idx] = 1
            h_prev = h
        return indices

```