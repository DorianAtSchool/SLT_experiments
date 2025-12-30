# Introduction

This repo contains my first experiments with Singular Learning Theroy, Local Learning Coefficient (LLC), grokking, and developmental interpretability. 

The main experiment is in /curriculum_learning and is described in detail below. It is still a work in progress but some interesting observations can already be seen. I aim to conduct more robust experiments in order to make more definitive conclusions and synthesize obsevrations into a blog post.

The rest of the repo contains other extra experiments (some work, some don't) reproducing existing papers from scratch (using PyTorch).

The long-term vision of the repo is to conduct novel experiments and form conclusive insights, however the short-term goal is simply to familiarize myself with developmental interpretability through novel but simple experiments.

# Curriculum Learning and LLC in Grokking

The main experiment explores the impact of curriculum learning versus direct training on the grokking behavior and Local Learning Coefficient (LLC) in modular arithmetic tasks, using PyTorch and the devinterp library by Timaeus.

## Overview
- **Goal:** Compare curriculum learning (progressively harder modular addition tasks) with direct training on the hardest task, analyzing LLC trajectories and grokking.
- **Notebook:** All code and analysis are in `devinterp/examples/curriculum_grokking_llc.ipynb`.
- **Reference:** Based on the original grokking experiment from [devinterp](https://github.com/timaeus-research/devinterp/blob/main/examples/grokking.ipynb).
- **Future Directions** It is unclear whether curriculum learning improves or worsens adaptability (either through memorization or grokking): when a previous model grokked, the subsequent models did not always show grokking, but when they did they (often) did so much quicker compared to direct training. Future, more specific experiments, will be done to verify whether curriculum learning encourages memorization vs. grokking, and further tune the conditions neccessary for grokking.

## Experiment Design
- **Curriculum Learning:**
  - Train a model sequentially on increasing modulus (prime) tasks.
  - At each stage, transfer model weights to the next modulus.
  - Track LLC and accuracy/loss at each stage.
- **Direct Training:**
  - Train a separate model directly on the hardest modulus.
  - Track LLC and accuracy/loss throughout training.

## Key Features
- **Reproducibility:** All random seeds are set for consistent results.
- **Device Management:** All tensors and models are placed on the correct device (CPU/GPU).
- **LLC Estimation:** (WIP) Uses devinterp's SGLD-based estimator to compute LLC for each checkpoint.
- **Visualization:** Plots accuracy, loss, and LLC trajectories for both approaches.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run `curriculum_learning/curriculum_grokking_llc.ipynb` in Jupyter or VS Code.
3. Follow the notebook sections:
   - Imports and setup
   - Experiment parameters
   - Model and utility functions
   - Dataset creation
   - Curriculum and direct training loops
   - LLC estimation
   - Plotting and analysis

## Observations / Results
It seems that curriculum learning can significanly decrease the number of steps to start grokking, and can accelerate grokking, when applied to a final modulus task that a model is known to have the capacity to grokk on. However, grokking doesnt seem to ever take place for many numbers or when modifiying seed and weight decay parameters in direct training or curriculum learning, and it is unclear whether curriculum learning improves or worsens adaptability (either through memorization or grokking): when a previous model grokked, the subsequent models did not always show grokking, but when they did they (often) did so much quicker compared to direct training.

- The notebook produces plots comparing curriculum and direct training in terms of accuracy, loss, and LLC.

## Files
- `devinterp/examples/curriculum_grokking_llc.ipynb`: Main experiment notebook
- `requirements.txt`: (if present) lists dependencies

## References
- [devinterp GitHub](https://github.com/timaeus-research/devinterp)
- Original grokking experiment: `examples/grokking.ipynb` in devinterp

## Contact
For questions or contributions, see the devinterp repository or open an issue.
