# Encoding

In machine learning, encoding refers to the process of transforming categorical data into a numerical format that can be easily understood and processed by machine learning algorithms. Since most algorithms require numerical input, encoding is a crucial step in preparing data for training models.

## One-Hot Encoding

One-Hot Encoding is a popular encoding technique used to convert categorical variables into a **binary matrix** representation. Each category is represented by a binary vector where only one element is "1" _(indicating the presence of that category)_ and all other elements are "0".

For example, consider a categorical variable "Color" with three categories: Red, Green, and Blue. The One-Hot Encoding for these categories would be:

| Color | Red | Green | Blue |
|-------|-----|-------|------|
| Red   |  1  |   0   |  0   |
| Green |  0  |   1   |  0   |
| Blue  |  0  |   0   |  1   |

## Python Implementation

```python title="encoding.py"
import numpy as np

class Encoding:
    @staticmethod
    def one_hot_encode(categories: list) -> np.ndarray:
        '''
        categories: List of categorical values (strings or integers)
        
        Returns:
            A 2D numpy array representing the one-hot encoded matrix
        '''
        unique_categories = list(set(categories))
        category_to_index = {category: index for index, category in enumerate(unique_categories)}
        
        one_hot_matrix = np.zeros((len(categories), len(unique_categories)))
        
        for i, category in enumerate(categories):
            index = category_to_index[category]
            one_hot_matrix[i, index] = 1
            
        return one_hot_matrix
```