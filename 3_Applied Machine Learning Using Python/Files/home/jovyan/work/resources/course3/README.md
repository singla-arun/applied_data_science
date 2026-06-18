# Multicollinearity — Concept Notes

## What is Multicollinearity?

When two or more features in a dataset are highly correlated — one can be predicted from another.

---

## Why It Hurts Model Performance

Multicollinearity does **not** reduce training accuracy. It causes:

| Problem | Effect |
|---|---|
| Coefficient instability | Small data changes → wildly different β values |
| Inflated variance | Standard errors blow up → confidence intervals useless |
| Wrong feature importance | Coefficients shrink/flip sign → feature selection breaks |
| Poor generalization | Unstable coefficients fail on new data outside training distribution |

**Root cause:** The system `Xβ = y` has infinitely many solutions. Optimizer picks one arbitrarily — that arbitrary choice does not generalize.

---

## Simple Example: House Price Prediction

Features: **Size (sqft)** and **Rooms**, where rooms = size / 250 (perfectly correlated).

| Size | Rooms | Price |
|------|-------|-------|
| 1000 | 4     | $200k |
| 1500 | 6     | $300k |
| 2000 | 8     | $400k |

### Why Multiple Solutions Exist

The model solves:

```
β₁×size + β₂×rooms = price
```

Substituting rooms = size/250:

```
β₁×1000 + β₂×4  = 200k
β₁×1500 + β₂×6  = 300k
β₁×2000 + β₂×8  = 400k
```

All 3 equations collapse to **one independent equation:**

```
β₁ + β₂/250 = 0.2
```

2 unknowns, 1 equation → infinite valid solutions:

- β₁ = 0.2, β₂ = 0
- β₁ = 0,   β₂ = 50
- β₁ = 0.1, β₂ = 25

All produce identical predictions on training data.

### Why Solutions Diverge on New Data

Training data only samples houses where rooms = size/250 — a 1D slice of 2D feature space.

New house: **2000 sqft, 5 rooms** (older home — off the diagonal).

```
β₁=0.2, β₂=0   → price = 0.2×2000 + 0×5     = $400k
β₁=0,   β₂=50  → price = 0×2000   + 50×5     = $250k
```

Same training accuracy. $150k difference on new data.

**Geometric intuition:** Each solution is a different plane in (size, rooms, price) space. All planes pass through training points on the diagonal. Off the diagonal — planes diverge.

---

## Fixes

| Method | How |
|---|---|
| Drop correlated features | VIF analysis — remove features with VIF > 5-10 |
| PCA | Combine correlated features into orthogonal components |
| Ridge Regression | L2 penalty shrinks unstable coefficients toward zero |
| Feature engineering | Replace correlated features with a single meaningful one |

---

# Missing Data — Concept Notes

## Why ML Models Can't Handle Missing Data

Most ML algorithms require a **fixed-size numeric vector** as input. A missing value breaks that contract.

---

## What Breaks Mathematically

**Distance-based (KNN, SVM, K-Means):**
```
distance(A, B) = √((a₁-b₁)² + (a₂-b₂)² + ...)
```
If a₂ is missing — undefined math. Can't compute.

**Linear models:**
```
ŷ = β₀ + β₁x₁ + β₂x₂
```
If x₂ is NaN → `β₂ × NaN = NaN`. Propagates through entire prediction.

**Tree models:** Split on feature value — `x > 5?` — NaN is neither yes nor no.

**Matrix operations (PCA, neural nets):** `XᵀX` blows up with NaN entries.

---

## Why Not Just Skip Missing Features?

Input shape must be **consistent** across all samples. Model trained on 10 features expects 10 features at prediction time — always.

Sample A: `[1, 2, 3]` vs Sample B: `[1, ?, 3]` → different shapes → can't process in same matrix operation.

---

## Exceptions — Models That Handle Missingness Natively

| Model | How |
|---|---|
| XGBoost / LightGBM | Learns which branch to send NaN values down |
| CatBoost | Treats NaN as separate category |
| Decision Trees (some impls) | Surrogate splits |

Even these need enough non-missing data to learn the pattern first.

---

## Standard Fixes

| Strategy | When to use |
|---|---|
| Mean/median imputation | Random missingness, numeric |
| Mode imputation | Categorical features |
| KNN imputation | Missingness correlated with other features |
| Add indicator column | Missingness itself is informative |
| Drop rows/columns | >50% missing — not worth imputing |
