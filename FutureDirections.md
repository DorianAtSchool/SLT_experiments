# Future Experiments & Questions

## Questions from Curriculum Learning

### 1. Does the Critical Threshold Scale?
- **Hypothesis:** The ~400 training pairs threshold might scale with task complexity
- **Experiments:**
  - Test different moduli (mod-32, mod-128, mod-256)
  - Test different operations (multiplication, composition)
  - See if threshold ∝ sqrt(modulus × input_range)
  - Investigate if threshold depends on model capacity

### 2. Can We Predict Grokking?
- **Question:** Can we use early LLC measurements to predict if/when grokking will occur?
- **Experiments:**
  - Track LLC in first 10% of training
  - Correlate early LLC patterns with eventual grokking
  - Develop early stopping criteria based on LLC
  - Test if LLC slope predicts grokking onset

### 3. Optimal Curriculum Design
- **Questions:**
  - Are 4 stages optimal, or would 8 smaller stages work better?
  - What's the ideal progression strategy? (linear, exponential, adaptive)
  - Should we adjust training time per stage based on LLC convergence?
- **Experiments:**
  - Fine-grained curriculum: [0,3], [0,7], [0,15], [0,31], [0,63]
  - Exponential growth: [0,7], [0,15], [0,63]
  - Adaptive: next stage triggered by LLC plateau

### 4. Curriculum for Different Model Sizes
- **Hypothesis:** Curriculum learning might benefit smaller models more than larger ones
- **Experiments:**
  - Test curriculum vs direct on models with hidden_size: 12, 24, 48, 96, 192
  - See if smaller models need curriculum to reach good performance
  - Investigate if larger models can skip early stages

## Questions from Finetuning

### 5. When Does Finetuning Help vs Hurt?
- **Key Question:** What conditions make finetuning beneficial?
- **Experiments:**
  - Systematic sweep of moduli sequences
  - Test different finetuning durations
  - Vary similarity between tasks (close moduli vs distant moduli)
  - Track when finetuning leads to catastrophic forgetting

### 6. Optimal Finetuning Strategy
- **Questions:**
  - Should we finetune on easy tasks first or hard tasks first?
  - Does task ordering matter?
  - How many finetuning stages are optimal?
- **Experiments:**
  - Compare: easy→hard vs hard→easy vs random order
  - Test 2, 3, 5, 10 finetuning stages
  - Measure final performance as function of sequence complexity

### 7. Transfer Without Catastrophic Forgetting
- **Question:** Can we maintain performance on previous tasks while learning new ones?
- **Experiments:**
  - Test continual learning techniques (EWC, replay buffers)
  - Multi-task training (train on all moduli simultaneously)
  - Compare to sequential finetuning

## LLC Methodology Questions

### 8. LLC Hyperparameter Sensitivity
- **Question:** Are our LLC estimates robust to hyperparameter choices?
- **Experiments:**
  - Full epsilon-beta sweep for each curriculum stage
  - Test if different stages need different hyperparameters
  - Validate convergence with longer chains (num_draws > 1000)
  - Compare SGLD vs other sampling methods

### 9. LLC as a Universal Metric?
- **Question:** Does LLC→test performance relationship hold for other tasks?
- **Experiments:**
  - Test on other algorithmic tasks (sorting, parsing, regex)
  - Test on vision tasks (MNIST, CIFAR-10 subsets)
  - Test on language tasks (grammar learning, simple NLP)
  - See if LLC threshold for generalization is task-dependent

### 10. What Does LLC Really Measure?
- **Theoretical Question:** Why does LLC (training-only) predict test performance?
- **Investigations:**
  - Correlate LLC with other complexity metrics (parameter norm, gradient norm)
  - Study relationship between LLC and loss landscape geometry
  - Investigate connection to information theory (minimum description length)
  - Explore if LLC captures "algorithmic complexity" of learned solution

## Architecture & Scaling

### 11. Architecture Impact on Grokking
- **Questions:**
  - Do different architectures (MLP, Transformer, Conv) show same grokking behavior?
  - How does depth vs width affect grokking threshold?
  - Does initialization strategy matter?
- **Experiments:**
  - Compare MLP, Transformer, ResNet on modular arithmetic
  - Sweep depth and width independently
  - Test different initializations (Xavier, He, orthogonal)

### 12. Scaling Laws for Grokking
- **Question:** How does grokking scale with model size and data size?
- **Experiments:**
  - Systematic sweep of (model_size, data_size) combinations
  - Plot grokking onset as function of scale
  - See if there are emergent thresholds
  - Compare to neural scaling laws literature

## Transfer Learning Deep Dive

### 13. Cross-Task Transfer
- **Question:** After grokking mod-64 addition, can the model quickly learn mod-64 multiplication?
- **Experiments:**
  - Train to grokking on addition, then transfer to multiplication
  - Test transfer to composite operations: (a+b)×(c+d) % 64
  - Measure if "general modular arithmetic" ability emerges

### 14. Few-Shot Learning After Curriculum
- **Question:** Does curriculum learning enable better few-shot adaptation?
- **Experiments:**
  - After Stage 3 (71% on [0,31]), how many Stage 4 examples are needed?
  - Compare curriculum→few-shot vs direct training on same data budget
  - Test if curriculum creates better "initialization" for new tasks

### 15. Negative Transfer
- **Question:** When does prior training hurt rather than help?
- **Experiments:**
  - Train on contradictory tasks (different moduli with same input range)
  - Measure interference between tasks
  - Identify conditions for negative transfer

## Data Efficiency

### 16. Can We Skip Stages?
- **Question:** After Stage 3 grokking, can we skip directly to harder tasks?
- **Experiments:**
  - Train Stage 1→3, then jump to mod-128 or mod-256
  - See if the learned algorithm generalizes to larger moduli
  - Test if we can "bootstrap" to arbitrary difficulty

### 17. Data Augmentation for Grokking
- **Question:** Can data augmentation accelerate grokking?
- **Experiments:**
  - Add synthetic examples (commutative: a+b = b+a)
  - Add derived examples (if (a+b)%64=c, then (a+b+64)%64=c)
  - Test if understanding of modular arithmetic structure helps

### 18. Active Learning for Curriculum
- **Question:** Can we adaptively select which examples to add next?
- **Experiments:**
  - Use LLC to identify when model is ready for harder examples
  - Select examples that maximally reduce LLC (most informative)
  - Compare to random curriculum progression

## Theoretical Questions

### 19. Equivalence to Human Learning?
- **Hypothesis:** Humans may show similar curriculum vs direct learning trade-offs
- **Investigations:**
  - Do humans learning programming show similar "critical mass" thresholds?
  - Do self-taught learners (analogous to direct training) explore higher complexity before simplifying?
  - Can we measure "human LLC" through problem-solving strategies?
  - Does structured education (curriculum) lead to different internal representations than self-teaching?

### 20. Loss Landscape Geometry
- **Question:** How do curriculum and direct training differ in loss landscape exploration?
- **Investigations:**
  - Visualize loss landscape using dimensionality reduction
  - Track optimization path for both approaches
  - Measure local minima quality (sharpness, basin size)
  - See if curriculum follows "easier" optimization path

### 21. Phase Transitions in Learning
- **Question:** Is grokking a true phase transition in the thermodynamic sense?
- **Investigations:**
  - Measure order parameters during grokking
  - Look for critical phenomena (diverging correlation length)
  - Connect to statistical physics of learning
  - Test if LLC shows critical slowing down before grokking

## Reproducibility & Robustness

### 22. Seed Sensitivity
- **Question:** How sensitive are our results to random seed?
- **Experiments:**
  - Run curriculum and direct training with 20+ seeds
  - Measure variance in grokking onset, final LLC, test accuracy
  - Identify conditions for reproducible grokking
  - Test if curriculum reduces seed sensitivity

### 23. Hyperparameter Robustness
- **Question:** Do results hold across different hyperparameter settings?
- **Experiments:**
  - Sweep learning rate, weight decay, batch size
  - Test different optimizers (Adam, SGD, AdamW)
  - Vary model architecture (hidden size, depth)
  - Identify robust vs brittle findings

## Meta-Learning

### 24. Learning to Learn
- **Question:** Can we meta-learn an optimal curriculum strategy?
- **Experiments:**
  - Train a meta-learner to predict optimal next task
  - Use RL to learn curriculum policy
  - Test if learned curriculum transfers to new domains
  - Compare to hand-designed curricula

### 25. Curriculum for Curriculum Design
- **Meta-Question:** What's the optimal way to discover optimal curricula?
- **Approach:**
  - Use LLC trajectories as feedback signal
  - Evolutionary algorithms for curriculum search
  - Multi-armed bandits for stage selection
  - Learn from successful transfer events
