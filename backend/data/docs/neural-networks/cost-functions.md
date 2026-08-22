# Cost Functions

Cost or Loss functions are mathematical functions that measure the difference between the predicted output of a NN and the actual target values. They quantify how well the NN is performing and guide the training process by providing feedback on the accuracy of the predictions.

## Mean Squared Error (MSE)

MSE is commonly used for regression problems. It calculates the average of the squared differences between the predicted and actual values.

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $ n $ is the number of data points
- $ y_i $ is the actual value
- $ \hat{y}_i $ is the predicted value

## Cross-Entropy Loss

Cross-Entropy Loss is commonly used for classification problems. It measures the difference between the predicted probability distribution and the actual distribution. It is designed to work perfectly with activation functions like Sigmoid and Softmax.

$$
CE = - \sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

Where:
- $ n $ is the number of classes
- $ y_i $ is the actual class label (1 for the correct class, 0 for others)
- $ \hat{y}_i $ is the predicted probability for class $ i $

### Binary Cross-Entropy

Used for binary classification problems (2 classes). It's a special case of Cross-Entropy Loss. For a single example, it is defined as:

$$
BCE = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

Where:
- $ y $ is the actual label (0 or 1)
- $ \hat{y} $ is the predicted probability of the positive class (1)

## Python Implementation

```python title="cost_functions.py"
import numpy as np

class Cost_Function:
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sq_errors = (y_true - y_pred) ** 2
        return np.mean(sq_errors)

    @staticmethod
    def ce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Adding a small value to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        ce_loss_ = -np.sum(y_true * np.log(y_pred), axis=1) # -> (m,)
        ce_loss = np.mean(ce_loss_)
        return ce_loss
    
    @staticmethod
    def bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Adding a small value to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        bce_loss_ = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        bce_loss = np.mean(bce_loss_)
        return bce_loss
    
    @staticmethod
    def ce_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Gradient of Cross-Entropy Loss with respect to predictions
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        grad = - (y_true / y_pred)
        return grad
    
    @staticmethod
    def bce_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Gradient of Binary Cross-Entropy Loss with respect to predictions
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        grad = - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        return grad
```

:::info Why epsilon and what is np.clip?

The `y_pred` values should be in the range (0, 1). To avoid taking the logarithm of zero (which is $-\infty$), we use a small constant value `epsilon` to clip the predicted probabilities. This ensures numerical stability during the computation of the Cross-Entropy Loss.

The `np.clip` function restricts the values in `y_pred` to be within the range `[epsilon, 1 - epsilon]`. This prevents any predicted probabilities from being exactly 0 or 1, which would lead to undefined logarithmic values.

:::