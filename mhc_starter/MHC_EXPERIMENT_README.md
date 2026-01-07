# mHC Grokking + LLC Experiment

## Overview

This experiment compares **standard MLP** vs **MLP with mHC (Manifold-Constrained Hyper-Connections)** on the modular addition grokking task.

**Paper**: [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek, Dec 2025)

**Implementation**: Based on [tokenbender/mHC-manifold-constrained-hyper-connections](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)

---

## What is mHC?

### Standard Residual Connection
```
x_{l+1} = x_l + F(x_l)
```

### Hyper-Connections (Unstable)
```
x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
```
Problem: Can cause training instability and signal explosion

### mHC (The Fix)
```
x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
```
**Key Constraint**: `H_res` is **doubly stochastic** (rows & cols sum to 1)

Achieved via **Sinkhorn-Knopp algorithm**:
- Iteratively normalizes rows and columns
- Projects onto Birkhoff polytope (doubly stochastic manifold)
- ~20 iterations for convergence

### Benefits

✅ **Training Stability** - Prevents signal explosion/collapse

✅ **Identity Mapping Preserved** - Like standard residuals but learnable

✅ **Better Scaling** - Tested at 3B, 9B, 27B parameters by DeepSeek

✅ **Low Overhead** - Only 6.7% additional training cost

✅ **Performance Gains** - +2.1% on BBH benchmark for 27B models

---

## Our Experiment

### Task: Modular Addition

Compute `(a + b) % 64` where `a, b ∈ [0, 63]`

- 4,096 total pairs
- 30% training, 70% test split (small training set for grokking)
- 100k training batches

### Models Compared

#### 1. Baseline MLP (Standard)
```python
x_left = linear1l(embed(a))
x_right = linear1r(embed(b))
x = x_left + x_right  # Simple addition
x = GELU(x)
output = linear2(x)
```

#### 2. MLP with mHC
```python
x_left = linear1l(embed(a))
x_right = linear1r(embed(b))
x = mHC(x_left, x_right)  # Doubly stochastic mixing!
x = GELU(x)
output = linear2(x)
```

### mHC Implementation

Our simplified `SimpleHyperConnection` layer:

```python
class SimpleHyperConnection(nn.Module):
    def __init__(self, dim, num_streams=2, mhc_num_iters=20, mhc_tau=0.05):
        super().__init__()
        # Initialize near-identity
        init_h_res = torch.full((2, 2), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)

    def forward(self, x_left, x_right):
        # Stack: [batch, 2, dim]
        x_stacked = torch.stack([x_left, x_right], dim=1)

        # Apply Sinkhorn-Knopp to get doubly stochastic matrix
        H_res = sinkhorn_log(self.H_res_logits,
                             num_iters=self.mhc_num_iters,
                             tau=self.mhc_tau)

        # Mix streams
        x_mixed = einsum(H_res[0], x_stacked, 's, b s d -> b d')
        return x_mixed
```

### Parameter Count

- **Baseline**: ~5,088 parameters
- **mHC**: ~5,092 parameters (+4 parameters = 2×2 mixing matrix)
- **Overhead**: +0.08% (negligible!)

---

## Research Questions

### 1. Does mHC Help with Grokking?

**Grokking** = delayed generalization where models first memorize (high train acc, low test acc), then suddenly generalize.

**Questions**:
- Does mHC grok faster?
- Does mHC achieve better final test accuracy?
- Does mHC have a smoother transition?

### 2. Does mHC Prevent Training Pathologies?

Our original experiments showed:
- 100% train accuracy, 0% test accuracy in early stages
- Severe overfitting

**Question**: Does mHC's stability prevent this?

### 3. How Does mHC Affect LLC?

**LLC (Local Learning Coefficient)** measures model complexity.

**Hypotheses**:
- **H1**: mHC has **lower LLC** (simpler due to doubly stochastic constraint)
- **H2**: mHC has **higher LLC** (more complex due to learnable mixing)
- **H3**: mHC has **different trajectory** (different path to solution)

### 4. Statistical Validation

- Run 3 seeds minimum
- T-test for final accuracy comparison
- Effect size (Cohen's d)
- Confidence intervals on all plots

---

## How to Run

### Quick Start

```bash
cd main_experiment
jupyter notebook grokking_mhc_experiment.ipynb
```

### Requirements

```bash
pip install torch devinterp scipy einops
```

### Expected Runtime

- **Training**: ~10 minutes per seed on GPU (3 seeds × 2 models = 60 min)
- **LLC Estimation**: ~30 minutes for seed 0 (sampled checkpoints)
- **Total**: ~1.5 hours on single GPU

### Hardware

- GPU recommended (CUDA compatible)
- CPU works but slower (~5x)
- Memory: ~4GB VRAM for mod-64 task

---

## Expected Outputs

### Plots Generated

1. **`baseline_vs_mhc_accuracy.png`** - Test accuracy comparison
2. **`generalization_gap.png`** - Train-test gap over time
3. **`llc_vs_accuracy.png`** - LLC trajectories with accuracy
4. **`llc_comparison.png`** - Direct LLC comparison

### Metrics Tracked

- Train/test accuracy
- Train/test loss
- Generalization gap
- LLC evolution
- Statistical significance

---

## Interpretation Guide

### Scenario 1: mHC Helps

**Observations**:
- mHC reaches higher final test accuracy
- mHC groks faster (earlier test accuracy increase)
- p-value < 0.05

**Interpretation**:
- Doubly stochastic constraint aids learning
- Stability prevents local minima
- **Recommendation**: Use mHC for grokking tasks!

### Scenario 2: mHC Hurts

**Observations**:
- mHC has lower final test accuracy
- mHC groks slower or not at all
- p-value < 0.05

**Interpretation**:
- Constraint is too restrictive for this task
- Simple addition works better than learned mixing
- **Recommendation**: Not beneficial for this architecture

### Scenario 3: No Difference

**Observations**:
- Similar final accuracy
- p-value > 0.05

**Interpretation**:
- Task too simple to benefit from mHC
- Overhead not worth it
- **Recommendation**: Use baseline (simpler)

### LLC Interpretation

**Lower LLC with mHC**:
- mHC learns a simpler solution
- Constraint reduces effective dimensionality
- More interpretable model

**Higher LLC with mHC**:
- mHC learns a more complex solution
- Mixing matrices add richness
- May be overfitting

**Different trajectory**:
- mHC takes a different path through parameter space
- Could indicate different algorithm discovered

---

## Limitations

### Computational

- Full multi-seed LLC estimation is expensive
- Only sampling every 10th checkpoint
- Only estimating LLC for seed 0

### Task-Specific

- Small modular arithmetic task
- May not generalize to other domains
- Limited to MLP architecture

### Implementation

- Simplified mHC (not full paper version)
- No kernel fusion optimizations
- No multi-GPU support

---

## Extensions

### Easy Extensions

1. **More seeds** - Run 5-10 seeds for stronger statistics
2. **Different moduli** - Try mod-32, mod-128
3. **Vary train fraction** - Test with 10%, 20%, 40% training data
4. **Hyperparameter sweep** - Try different mHC tau, num_iters

### Advanced Extensions

1. **Full mHC** - Use complete implementation with H_pre, H_post
2. **Multi-layer mHC** - Add mHC between all layers
3. **True curriculum** - Use expanding input ranges
4. **Deeper networks** - Test 4-layer, 6-layer MLPs
5. **Other tasks** - Polynomial evaluation, permutations, etc.

### Research Directions

1. **mHC + Curriculum Learning** - Do they combine synergistically?
2. **LLC Decomposition** - Attribute LLC to specific components
3. **Interpretability** - Visualize learned H_res matrices
4. **Scaling Laws** - How does mHC affect scaling behavior?

---

## Troubleshooting

### Import Errors

```bash
# Install einops
pip install einops

# Install devinterp
pip install devinterp
```

### CUDA Out of Memory

```python
# Reduce batch size
params.batch_size = 64  # Default: 128

# Sample fewer checkpoints for LLC
for i in range(0, len(checkpoints), 20):  # Every 20th instead of 10th
```

### LLC Warnings

```
UserWarning: taking more draws than burn-in steps
```

**Solution**: Increase num_draws or decrease burn-in (expected for quick experiments)

### NaN Losses

If you see NaN losses with mHC:

```python
# Reduce learning rate
params.lr = 0.0005  # Default: 0.001

# Increase mHC tau (less aggressive)
mhc_tau = 0.1  # Default: 0.05
```

---

## References

### Papers

- **mHC**: Manifold-Constrained Hyper-Connections (DeepSeek, 2025) - [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
- **Hyper-Connections**: Original HC paper - [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)
- **Grokking**: Power et al. (2022) - [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- **SLT & LLC**: Singular Learning Theory for neural networks

### Code

- **mHC Implementation**: [github.com/tokenbender/mHC-manifold-constrained-hyper-connections](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)
- **DevInterp**: [github.com/timaeus-research/devinterp](https://github.com/timaeus-research/devinterp)

### Additional Resources

- **DeepSeek Blog**: Announcements and technical details
- **Sinkhorn-Knopp Algorithm**: Doubly stochastic projection method
- **Birkhoff Polytope**: Mathematical manifold of doubly stochastic matrices

---

## Contributing

Found an issue or have improvements?

1. Test your changes
2. Document new features
3. Share results in Issues or PRs

---

## License

This experiment follows the licenses of:
- mHC implementation: Apache 2.0
- DevInterp: MIT
- Your original code: (specify)

---

**Created**: 2026-01-03
**Author**: Based on tokenbender's mHC implementation
**Contact**: See repository
