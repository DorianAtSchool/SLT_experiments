# Curriculum Learning Approaches for Modular Addition

## Three Different Approaches Compared

### ❌ Approach 1: Different Moduli (Original - NOT curriculum learning)

**What you did originally:**
```
Stage 1: (a+b) % 8  where a,b ∈ [0,7]
Stage 2: (a+b) % 16 where a,b ∈ [0,15]
Stage 3: (a+b) % 32 where a,b ∈ [0,31]
Stage 4: (a+b) % 64 where a,b ∈ [0,63]
```

**Problems:**
- ❌ Different functions! `(5+7) % 8 = 4` but `(5+7) % 16 = 12`
- ❌ Stage 1 is NOT a subset of stage 2
- ❌ Embedding layer size changes (8 → 16 → 32 → 64)
- ❌ Output layer size changes
- ❌ Only ~40-50% of weights transfer
- ❌ This is **transfer learning**, not curriculum learning

**When to use:**
- Never for curriculum learning claims
- Could be interesting for studying transfer learning across related tasks

---

### ✅ Approach 2: Fixed Modulus, Expanding Inputs (TRUE curriculum)

**Implementation:** `true_curriculum_learning.ipynb`

```
Stage 1: (a+b) % 64 where a,b ∈ [0,7]    → 64 pairs
Stage 2: (a+b) % 64 where a,b ∈ [0,15]   → 256 pairs
Stage 3: (a+b) % 64 where a,b ∈ [0,31]   → 1,024 pairs
Stage 4: (a+b) % 64 where a,b ∈ [0,63]   → 4,096 pairs
```

**Advantages:**
- ✅ Same function throughout!
- ✅ Stage 1 pairs are a TRUE subset of stage 2 pairs
- ✅ Fixed vocabulary (64 tokens always)
- ✅ Same model architecture throughout
- ✅ 100% weight transfer between stages
- ✅ This IS curriculum learning

**Example:**
- Stage 1: Learn `(3,5) → 8`, `(7,2) → 9`, etc. (64 pairs)
- Stage 2: Keep all stage 1 pairs + add `(12,8) → 20`, etc. (256 total pairs)
- The model already knows stage 1 pairs perfectly, just adds new ones

**Why it works:**
- Progressive difficulty: more pairs to memorize
- Truly builds on previous knowledge
- No catastrophic forgetting possible (old pairs still in training set)

**Potential issues:**
- May be "too easy" - model might just memorize without generalizing
- Final stage is same as direct training (same data, same task)
- Benefit only comes from learning order, not task structure

---

### ⚠️ Approach 3: Fixed Modulus, Expanding BOTH Inputs and Outputs

**Alternative curriculum:**
```
Stage 1: a,b ∈ [0,7],  but only predict outputs in [0,15]  (restrict loss)
Stage 2: a,b ∈ [0,15], but only predict outputs in [0,31]  (restrict loss)
Stage 3: a,b ∈ [0,31], but only predict outputs in [0,63]  (restrict loss)
Stage 4: a,b ∈ [0,63], predict all outputs [0,63]         (full task)
```

**Implementation:**
- Use masked loss that only penalizes certain output classes
- Gradually "unlock" more output neurons

**Advantages:**
- ✅ Still true subsets
- ✅ Fixed vocab size
- ✅ 100% weight transfer
- ✅ May avoid memorization (model has to generalize to new outputs)

**Disadvantages:**
- ⚠️ More complex to implement
- ⚠️ Harder to interpret
- ⚠️ Output masking feels artificial

---

## Comparison Table

| Aspect | Original (Different Moduli) | TRUE Curriculum (Expanding Inputs) | Masked Outputs |
|--------|---------------------------|-----------------------------------|----------------|
| **Same task?** | ❌ No (different moduli) | ✅ Yes (always % 64) | ✅ Yes |
| **True subsets?** | ❌ No | ✅ Yes | ✅ Yes |
| **Weight transfer** | ~40-50% | 100% | 100% |
| **Vocab size** | Changes | Fixed (64) | Fixed (64) |
| **Model resize?** | Yes | No | No |
| **Is curriculum?** | No (transfer learning) | Yes | Yes |
| **Difficulty source** | Different operation | More pairs | More pairs + outputs |

---

## Recommended Approach

**For studying curriculum learning:** Use Approach 2 (Expanding Inputs)

**Rationale:**
1. Mathematically correct - true curriculum learning
2. Simple and interpretable
3. No resizing headaches
4. 100% weight transfer
5. Can directly compare to direct training (same final task)

---

## Example Code Differences

### Original (Wrong):
```python
# Different modulus at each stage
CURRICULUM_NUMS = [8, 16, 32, 64]

for p in CURRICULUM_NUMS:
    dataset = make_dataset(p)  # All pairs in [0,p)
    # Train model with vocab_size=p
    # Transfer weights (partially fails due to size mismatch)
```

### TRUE Curriculum (Correct):
```python
# Fixed modulus, expanding input ranges
MODULUS = 64
INPUT_RANGES = [7, 15, 31, 63]  # Max values for a,b

for max_val in INPUT_RANGES:
    dataset = make_curriculum_dataset(max_val, modulus=MODULUS)
    # a,b ∈ [0, max_val], compute (a+b) % MODULUS
    # Train model with vocab_size=64 (always!)
    # Transfer ALL weights (100% transfer)
```

---

## Verification

To verify you have true curriculum learning:

```python
# After creating datasets
stage1_pairs = set((x[0], x[1]) for x, _ in stage1_data)
stage2_pairs = set((x[0], x[1]) for x, _ in stage2_data)

# This should be TRUE for curriculum learning
assert stage1_pairs.issubset(stage2_pairs), "Stage 1 must be subset of Stage 2!"

# This should also be TRUE
assert all((a+b) % MODULUS == label for (a,b), label in stage1_data), "Wrong modulus!"
assert all((a+b) % MODULUS == label for (a,b), label in stage2_data), "Wrong modulus!"
```

---

## Research Questions to Explore

With true curriculum learning (Approach 2), you can now ask:

1. **Does curriculum help?**
   - Compare final test accuracy: curriculum vs direct
   - With equalized training budgets!

2. **Does curriculum change LLC trajectories?**
   - Do models learn simpler solutions when using curriculum?
   - Does LLC increase or decrease compared to direct training?

3. **Generalization patterns?**
   - Does curriculum lead to better generalization on held-out pairs?
   - Does the model find different algorithmic solutions?

4. **Optimal curriculum design?**
   - Is [7,15,31,63] better than [3,7,15,31,63] (more stages)?
   - What about non-uniform spacing like [7,11,23,63]?

5. **Forgetting?**
   - Do models forget early stage pairs when learning later stages?
   - (They shouldn't, since all pairs remain in training set!)

---

## Implementation Files

1. **`true_curriculum_learning.ipynb`** - Correct curriculum learning implementation
2. **`curriculum_grokking_llc_corrected.ipynb`** - Fixed version of original (still transfer learning, not curriculum)
3. **`curriculum_grokking_llc.ipynb`** - Original flawed version (for reference)

---

## Conclusion

**Your original experiment was testing transfer learning across related tasks**, not curriculum learning. While interesting, it cannot make claims about curriculum learning specifically.

**To study true curriculum learning**, use the approach in `true_curriculum_learning.ipynb` where:
- All stages compute the same function
- Earlier stages are proper subsets
- Models can fully transfer all learned knowledge

This allows you to cleanly test whether learning order matters when the task stays constant but difficulty increases.
