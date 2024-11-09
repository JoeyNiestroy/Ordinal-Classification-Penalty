# Nominal-Classification-Penalty
A proposed KL-div based penalty to improve the performance of nominal classification tasks with NN backends with minimal comp costs

# KL Divergence Penalty for Nominal Classifications in Deep Learning

This repository contains an implementation of an approach that adds a Kullback-Leibler (KL) divergence penalty to the standard Cross-Entropy loss function for nominal classification tasks. This approach aims to improve performance by enforcing a smoother probability distribution over classes, effectively guiding the model towards more informed class predictions.

## Overview

In traditional classification tasks, the model is typically trained with a Cross-Entropy loss function, while this approach works well, it treats each class independently without incorporating any additional structure or inter-class relationships. In practice nominal predictions have underperformed with this base approach, see CORN loss and similar approaches for how researchers have tried to combat this issue. Below is the approach I've designed for this issue



### Loss Formulation
Let:
- $\mathbf{p_s}$ represent the target distribution (a Gaussian-like distribution centered on the correct class with $Var(s)$ ),
- $\mathbf{q}$ represent the model’s predicted probabilities after softmax,
- $N$ represent the number of classes.

The KL divergence component is given by:


$D_{KL}(\mathbf{p_s} \| \mathbf{q}) = \sum_{i=1}^{N} p_i \log\left(\frac{p_i}{q_i}\right)$


where $p_i$ and $q_i$ are the probabilities for class $i$ from the Gaussian target distribution and the model’s predictions, respectively.

The total loss function $\mathcal{L}$ used for training is a combination of Cross-Entropy and KL divergence:


$\mathcal{L} = \text{CrossEntropy}(\mathbf{q}, \mathbf{y}) + \alpha D_{KL}(\mathbf{p} \| \mathbf{q})$

where:
-  $\mathbf{y}$ is the true class label,
- $\alpha$ is a hyperparameter controlling the weight of the KL divergence penalty.

### Computational Complexity and Space Requirements
For a batch size of $B$ and a class size of $K$

Computational: $O(B \cdot K)$

Space: $O(B \cdot K)$

Both are linear. It should be noted the Gaussian data could be computed at time of inference to save memory if need be. 

## Experimental Section

### Code Explanation

Below is a breakdown of the key parts of the code used in this experiment.

### Data
The UTKFace dataset was used for an age estimation task formulated as a nominal classification task

### Model and Training

See the attached .py files for specfics on the models and training hyperparameters. The base model is a very simple CNN based model. Both control and KL models were trained for 50 epochs at a fixed lr and testing on a subset of clean data for MSE

### Gaussian Target Distribution

To construct a smoother target distribution, a Gaussian mound is created around the correct class label, assigning a probability to each class based on its distance from the correct class.

```python
import numpy as np

def create_gaussian_mound(num_classes, correct_class, sigma=1):
    x = np.arange(num_classes)
    target_probs = np.exp(-0.5 * ((x - correct_class) / sigma) ** 2)
    target_probs /= np.sum(target_probs)  # Normalize to sum to 1
    return target_probs
```

### KL Divergence Function

The KL Divergence function calculates the divergence between the Gaussian mound distribution and the model's softmax probabilities for the given logits.

```python
def kl_divergence(p, q):
    p = p + 1e-8  # Adding small epsilon to avoid log(0)
    return torch.sum(p * torch.log(p / q))
```

## Results and Observations


| Metric                         | Cross-Entropy (Base) | Cross-Entropy + KL Divergence Penalty |
|--------------------------------|----------------------|---------------------------------------|
| **Mean MSE on Clean Data**          | 188.38 ± x       | 169.61 ± x                        |
| **Confidence Interval (95%)**  | *     | *                       |

Confidence intervals will added once I'm able to re-run experiement on suitable number of iterations. For now we can see a minor improvment on the performance over our control model with base cross entropy which fingers crossed winds up being significant. 

It should be noted that neither model was optimized for the task through any hyperparemter searching, this experimental result should be taken as a proof of concept. 

