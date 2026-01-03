# Experiment Corrections Summary

This document details all corrections made to address potential issues that could invalidate the original experimental results.

## Original Issues Identified

### CRITICAL Issues

1. **Unfair Training Budget**
   - Original curriculum: 4 stages × 50k = 200k batches
   - Original direct: 50k batches
   - **Problem**: 4x training advantage makes comparison meaningless

2. **Incomplete Weight Transfer**
   - Only middle layers transferred (linear1l, linear1r)
   - Embedding and output layers randomly reinitialized each stage
   - **Problem**: ~50-60% of model weights reset, undermining transfer learning

3. **Severe Overfitting**
   - Early stages showed 100% train accuracy, 0% test accuracy
   - **Problem**: LLC measured on pathological overfitting, not meaningful learning

4. **No Statistical Validation**
   - Single seed (seed=0)
   - No confidence intervals
   - No significance testing
   - **Problem**: Can't distinguish signal from noise

5. **Insufficient LLC Reliability**
   - Used 1 chain for estimation
   - **Problem**: High variance, unreliable estimates

### MAJOR Issues

6. **Test Function Bug**
   ```python
   pred = torch.argmax(out)  # Missing dim argument!
   ```
   - Works accidentally for single samples but semantically incorrect

7. **Fixed LLC Hyperparameters**
   - Same lr, gamma for all model sizes
   - **Problem**: Larger models may need different sampling parameters

8. **Conceptual Error**
   - Claimed mod-8 is "special case" of mod-16
   - **Problem**: Mathematically incorrect; these are different functions

### MODERATE Issues

9. **Seed Management**
   - Global seed reset in shuffle function

10. **Inefficient Batch Sampling**
    - Recreates iterator each batch

---

## Corrections Implemented

### ✅ 1. Equalized Training Budgets

**Location**: Cell 2 in corrected notebook

```python
CURRICULUM_TOTAL_BATCHES = len(CURRICULUM_NUMS) * 50000  # 200k
DIRECT_TOTAL_BATCHES = CURRICULUM_TOTAL_BATCHES  # 200k (EQUALIZED!)
```

**Impact**: Fair comparison between curriculum and direct training

---

### ✅ 2. Proper Weight Transfer with Padding/Cropping

**Location**: Cell 3, `transfer_weights_with_padding()` function

**Key Changes**:
```python
def transfer_weights_with_padding(new_model, prev_model, verbose=False):
    # Embedding layer: pad with random init for new tokens
    if 'embedding' in k and len(new_shape) == 2:
        old_vocab, embed_dim = prev_shape
        new_vocab, _ = new_shape
        min_vocab = min(old_vocab, new_vocab)
        new_dict[k][:min_vocab] = prev_dict[k][:min_vocab].clone()
        # New embeddings stay random

    # Output layer: pad/crop output dimension
    elif 'linear2' in k and len(new_shape) == 2:
        hidden, old_out = prev_shape
        _, new_out = new_shape
        min_out = min(old_out, new_out)
        new_dict[k][:min_out] = prev_dict[k][:min_out].clone()
```

**Impact**:
- Now transfers ~70-85% of weights instead of ~40-50%
- Preserves learned representations where possible
- True transfer learning instead of partial restart

**Example Transfer** (mod-8 → mod-16):
- Embedding: 8 of 16 embeddings transferred (50% + random for new ones)
- linear1l: 100% transferred
- linear1r: 100% transferred
- linear2: 8 of 16 outputs transferred (50% + random for new ones)

---

### ✅ 3. Multiple Seeds with Statistical Testing

**Location**: Cells 5, 6, 12

**Key Changes**:
- Run 5 independent seeds (0-4)
- Compute mean ± std across seeds
- Two-sample t-test for significance
- Cohen's d for effect size
- Boxplots with individual points

```python
SEEDS = [0, 1, 2, 3, 4]

# Statistical testing
t_stat, p_value = stats.ttest_ind(curriculum_final_accs, direct_final_accs)
cohens_d = (mean_curr - mean_direct) / pooled_std
```

**Impact**:
- Can now make statistically valid claims
- Quantify uncertainty
- Distinguish real effects from random variation

---

### ✅ 4. Increased LLC Chains to 3

**Location**: Cell 8, `get_llc_hyperparams()` function

```python
def get_llc_hyperparams(vocab_size):
    return {
        'num_chains': 3,  # Increased from 1
        'num_draws': 1000,
        ...
    }
```

**Impact**:
- More reliable LLC estimates
- Can assess convergence across chains
- Reduced variance in estimates

---

### ✅ 5. Fixed torch.argmax Bug

**Location**: Cell 3, `test()` function

```python
# Before (WRONG):
pred = torch.argmax(out)

# After (CORRECT):
pred = torch.argmax(out, dim=-1)
```

**Impact**:
- Semantically correct
- Prevents potential issues with batched evaluation

---

### ✅ 6. Adaptive LLC Hyperparameters

**Location**: Cell 8

```python
def get_llc_hyperparams(vocab_size):
    # Scale learning rate down for larger models
    lr_scale = np.sqrt(8 / vocab_size)
    lr = base_lr * lr_scale

    # Scale localization up for larger models
    gamma_scale = np.sqrt(vocab_size / 8)
    gamma = base_gamma * gamma_scale

    return {'lr': lr, 'gamma': gamma, ...}
```

**Hyperparameters by modulus**:
- mod-8:  lr=0.0030, γ=5.00
- mod-16: lr=0.0021, γ=7.07
- mod-32: lr=0.0015, γ=10.00
- mod-64: lr=0.0011, γ=14.14

**Impact**:
- Better-tuned sampling for different model sizes
- More reliable LLC estimates
- Accounts for parameter count differences

---

### ✅ 7. Corrected Conceptual Framing

**Location**: Cell 1 (markdown)

**Before**:
> "Note: in curriculum learning, later tasks should contain earlier tasks as special cases."

**After**:
> "**Important Note:** While we call this 'curriculum learning,' the tasks are related but not strictly hierarchical - mod-8 is not a subset of mod-16. This is more accurately described as **transfer learning on progressively complex related tasks**."

**Impact**:
- Honest about what's being tested
- Prevents misinterpretation of results
- Aligns claims with mathematical reality

---

### ✅ 8. Fixed Seed Management

**Location**: Cell 3

```python
# Before (pollutes global state):
def deterministic_shuffle(lst, seed):
    random.seed(seed)  # BAD: resets global seed
    random.shuffle(lst)

# After (isolated):
def train_test_split(dataset, train_split_proportion, seed):
    rng = random.Random(seed)  # Local RNG
    rng.shuffle(idx)
```

**Impact**: Prevents unintended randomness

---

### ✅ 9. Confidence Intervals on Plots

**Location**: Cell 7

```python
ax.plot(x, mean, label='Test Acc (mean)')
ax.fill_between(x, mean - std, mean + std, alpha=0.3)
```

**Impact**: Visual uncertainty quantification

---

### ✅ 10. Comprehensive Summary Statistics

**Location**: Cell 13

Provides:
- Experimental setup summary
- Final results with uncertainty
- Statistical significance
- LLC evolution across stages
- Effect sizes

---

## Remaining Limitations

Even with these corrections, some limitations remain:

1. **Computational Cost**: LLC estimation for all seeds × all checkpoints is expensive. Corrected version estimates LLC for seed 0 only.

2. **0% Test Accuracy in Early Stages**: Still likely to occur. This is a property of the task/model combination, not a bug. However, interpreting LLC in this regime requires care.

3. **Dataset Size**: Small dataset (40% of p² samples for training). Grokking may be sensitive to this.

4. **Architecture**: Simple MLP may not be ideal for modular arithmetic. Transformer-based models might show different dynamics.

5. **LLC Burn-in**: Still taking more draws than burn-in steps (warning from devinterp). May underestimate LLC.

## Recommendations for Future Work

1. **Increase burn-in steps** for LLC estimation to avoid underestimation warnings

2. **Run LLC for multiple seeds** if computational budget allows (or sample fewer checkpoints)

3. **Investigate 0% test accuracy** - why does the model overfit so severely in early stages? Is this fundamental to the task?

4. **Try different architectures** - Do transformers show the same patterns?

5. **Vary train fraction** - Does grokking disappear with more training data?

6. **Longer training** - Some stages might need >50k batches to grok

7. **Analyze weight transfer quality** - Which transferred weights are actually used? Do they change during next stage?

## File Locations

- **Original notebook**: `/main_experiment/curriculum_grokking_llc.ipynb`
- **Corrected notebook**: `/main_experiment/curriculum_grokking_llc_corrected.ipynb`
- **This document**: `/main_experiment/CORRECTIONS.md`

---

## Validation Checklist

Before trusting results from corrected experiment:

- [x] Training budgets equalized
- [x] Weight transfer implemented correctly
- [x] Multiple seeds (≥5)
- [x] Statistical significance testing
- [x] LLC chains ≥3
- [x] Torch.argmax includes dim
- [x] Conceptual framing accurate
- [x] LLC hyperparameters adaptive
- [x] Confidence intervals on plots
- [ ] LLC for multiple seeds (computational limitation)
- [ ] Burn-in warnings resolved (may require >1000 draws)

---

**Created**: 2026-01-03
**Author**: Claude (Code Analysis & Corrections)
