# Polynomial Regression, Feature Transformations, ReLU Networks, and XOR – Consolidated Notes

## 1. What is Polynomial Regression?

Polynomial regression is **ordinary linear regression applied to transformed features**.

Instead of fitting:

$$y = \beta_0 + \beta_1 x$$

we create additional features such as:

$$x^2, \; x^3, \; x^4, \ldots$$

and fit:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \cdots$$

Although the relationship between $x$ and $y$ is nonlinear, the model is still called **linear regression** because it remains linear in the coefficients $\beta$.

---

## 2. Why Do We Need Polynomial Regression?

Many real-world relationships are not straight lines.

Examples:

- House price vs lot size
- Fuel efficiency vs speed
- Sales vs advertising spend
- Population growth

A linear model $y = \beta_0 + \beta_1 x$ can only learn straight-line relationships.

Polynomial regression allows the model to learn curved relationships.

---

## 3. Example

Suppose the true relationship is $y = x^2$.

| $x$ | $y$ |
| --- | --- |
| -2  | 4   |
| -1  | 1   |
| 0   | 0   |
| 1   | 1   |
| 2   | 4   |

A straight line cannot fit this perfectly.

Create a new feature $z = x^2$, then fit:

$$y = \beta_0 + \beta_1 x + \beta_2 z$$

The perfect solution is $\beta_0 = 0, \; \beta_1 = 0, \; \beta_2 = 1$, giving $y = x^2$.

---

## 4. Key Insight: Feature Transformation

Polynomial regression works by transforming the input space.

Original feature: $x$

Transformed features: $[x, \; x^2, \; x^3, \ldots]$

The algorithm itself does not change — only the features change.

---

## 5. Geometric Interpretation

### Original Space

The relationship $y = x^2$ appears as a curved parabola. A straight line cannot fit it perfectly.

### Transformed Space

Create $z = x^2$. The data now lives in $(x, \; z, \; y)$.

For all points: $y = z$, which is a **plane**.

The nonlinear relationship becomes linear in the transformed space.

Polynomial regression finds a hyperplane in this higher-dimensional space.

---

## 6. Important Clarification About the Hyperplane

A common misconception:

> If the fitted plane is $y = z$, does that mean $x = 0$?

No. The plane is $y = z$, not $x = 0$. It extends through all values of $x$.

The coefficient of $x$ is zero because once $z = x^2$ is known, $x$ contributes no additional information for predicting $y$.

The coefficient being zero does **not** mean the feature itself is zero.

---

## 7. Polynomial Features

### Degree 2

Features: $1, \; x, \; x^2$

$$y = \beta_0 + \beta_1 x + \beta_2 x^2$$

### Degree 3

Features: $1, \; x, \; x^2, \; x^3$

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$$

### Degree $n$

Features: $1, \; x, \; x^2, \ldots, x^n$

$$y = \sum_{i=0}^{n} \beta_i x^i$$

---

## 8. Multiple Variables

Suppose:
- Lot Size = $L$
- Property Tax = $T$

Degree-2 polynomial features: $L, \; T, \; L^2, \; T^2, \; LT$

$$\text{Price} = \beta_0 + \beta_1 L + \beta_2 T + \beta_3 L^2 + \beta_4 T^2 + \beta_5 LT$$

---

## 9. Interaction Terms

An interaction term is the product of two variables, e.g. $LT$.

Interaction terms allow the effect of one variable to depend on another.

Without interaction — effect of lot size is constant:

$$\text{Price} = \beta_0 + \beta_1 L + \beta_2 T$$

With interaction — effect of lot size changes as taxes change:

$$\text{Price} = \beta_0 + \beta_1 L + \beta_2 T + \beta_3 LT$$

---

## 10. Basis Functions

Polynomial features are examples of **basis functions** — transformations of an original feature into a new feature.

Examples: $x^2$, $x^3$, $\log(x)$, $\sin(x)$, $e^x$

General model:

$$y = \beta_0 + \beta_1 \phi_1(x) + \beta_2 \phi_2(x) + \cdots$$

where $\phi_i$ are basis functions.

---

## 11. Why Is It Still Called Linear Regression?

Because the model is **linear in the coefficients**.

$$y = \beta_0 + \beta_1 x + \beta_2 x^2$$

The feature $x^2$ is nonlinear. The coefficient $\beta_2$ appears linearly.

Linear regression requires linearity in the **parameters**, not in the inputs.

---

## 12. Overfitting Risk

Higher-degree polynomials are very flexible:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_{20} x^{20}$$

Advantages: can fit complex patterns.

Disadvantages: fits noise, poor generalization, high variance — this is **overfitting**.

---

## 13. Polynomial Regression vs Neural Networks

Both methods solve the same problem: transform data into a representation where a linear model works.

### Polynomial Regression

Human designs the features:

$$x \rightarrow [x, \; x^2, \; x^3]$$

The transformation is fixed. The model only learns coefficients.

### Neural Networks

The network learns the transformation itself. A neuron computes:

$$h = \text{ReLU}(Wx + b)$$

where

$$\text{ReLU}(z) = \max(0, z)$$

The weights $W$ are learned from data.

---

## 14. Why ReLU Is Important

Without ReLU:

$$h = W_1 x + b_1$$
$$y = W_2 h + b_2$$

Substituting gives $y = Ax + c$ — still a linear model. Stacking linear layers does not increase expressive power.

With ReLU:

$$h = \text{ReLU}(W_1 x + b_1)$$

the model becomes nonlinear and can learn much more complex relationships.

---

## 15. Can ReLU Learn $x^2$ and $x^3$?

A single ReLU cannot represent $x^2$ or $x^3$ exactly.

However, many ReLU neurons can approximate these functions arbitrarily well.

A deep network learns a **piecewise-linear approximation** to a smooth nonlinear function — similar to approximating a curve using many short straight line segments.

---

## 16. Polynomial Features vs Learned Features

**Polynomial regression:**

```
Input
  |
Create x²
Create x³
Create x₁x₂
  |
Linear Model
```

**Neural network:**

```
Input
  |
Learn Features
  |
Learn Better Features
  |
Linear Output Layer
```

Polynomial regression learns weights. Neural networks learn both the **features** and the **weights**.

---

## 17. What is XOR?

XOR (Exclusive OR) outputs 1 when exactly one input is 1.

| $x_1$ | $x_2$ | XOR |
| ----- | ----- | --- |
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

---

## 18. Why XOR Is Important

The positive examples $(0,1)$ and $(1,0)$ lie on opposite corners.

The negative examples $(0,0)$ and $(1,1)$ lie on the other corners.

No single line can separate them — XOR is **not linearly separable**.

---

## 19. XOR is a Nonlinear Interaction

XOR depends on the combination of variables.

Neither $x_1$ nor $x_2$ alone determines the output. The relationship depends on their **interaction**.

---

## 20. Polynomial Features Can Solve XOR

Add the interaction feature $x_1 x_2$.

Feature vector: $[x_1, \; x_2, \; x_1 x_2]$

This lifts the data into a higher-dimensional space where a plane can separate the classes — the same principle as polynomial regression.

---

## 21. How ReLU Learns XOR

A hidden layer learns new features:

$$h_1 = \text{ReLU}(x_1 + x_2 - 0.5)$$
$$h_2 = \text{ReLU}(x_1 + x_2 - 1.5)$$

These transform the original coordinates into a new representation where the points become linearly separable.

The final layer then draws a linear decision boundary.

---

## 22. Why Hidden Layers Matter

A single perceptron cannot solve XOR.

A neural network with a nonlinear activation and a hidden layer can transform the data into a space where XOR becomes linearly separable.

This was one of the earliest demonstrations of the power of multilayer neural networks.

---

## 23. Relationship Between Polynomial Regression, Kernel Methods, and Neural Networks

All three use the same fundamental idea: transform data into a feature space where a linear model works.

| Method                | Who Creates Features? |
| --------------------- | --------------------- |
| Polynomial Regression | Human                 |
| Kernel SVM            | Kernel Function       |
| Neural Network        | Learned from Data     |

---

## Summary

**Polynomial Regression:**
- Linear regression on transformed features
- Features may include $x^2, x^3, x_1 x_2$
- Linear in coefficients, nonlinear in inputs
- Finds a hyperplane in transformed feature space

**Neural Networks:**
- Learn feature transformations automatically
- ReLU introduces nonlinearity
- Hidden layers create new representations
- Can approximate $x^2$, $x^3$, XOR, and much more

**XOR:**
- Not linearly separable
- Requires interaction between variables
- Solved using polynomial features or hidden-layer neural networks
- Demonstrates why nonlinear feature transformations are necessary
