# Introduction

This repo contains my first experiments with Singular Learning Theory, Local Learning Coefficient (LLC), grokking, and developmental interpretability. 

The main experiment is in /main_experiment and is described in detail below. It is still a work in progress but some interesting observations can already be seen. I aim to conduct more robust experiments in order to make more definitive conclusions and synthesize observations into a blog post.

The rest of the repo contains other extra experiments (some work, some don't) reproducing existing papers from scratch (using PyTorch).

The long-term vision of the repo is to conduct novel experiments and form conclusive insights, however the short-term goal is simply to familiarize myself with developmental interpretability through novel but simple experiments.


# Curriculum Learning, Finetuning, and LLC in Grokking

The main experiment explores the impact of curriculum learning versus direct training on the grokking behavior and Local Learning Coefficient (LLC) in modular arithmetic tasks, using PyTorch and the devinterp library by Timaeus.


## Overview
- **Goal:** Compare three approaches—curriculum learning (progressively harder modular addition tasks), finetuning (sequential training on unrelated moduli), and direct training on the hardest task—analyzing LLC trajectories and grokking.
- **Notebook:** All code and analysis are in `main_experiment/curriculum_grokking_llc.ipynb`.
- **visuals** All visuals are in /results.
- **Reference:** Based on the original grokking experiment from [devinterp](https://github.com/timaeus-research/devinterp/blob/main/examples/grokking.ipynb).
- **Future Directions:** It is unclear whether curriculum learning or finetuning improves or worsens adaptability (either through memorization or grokking): when a previous model grokked, the subsequent models did not always show grokking, but when they did they (often) did so much quicker compared to direct training. Future, more specific experiments, will be done to verify whether curriculum learning or finetuning encourages memorization vs. grokking, and further tune the conditions necessary for grokking.


## Experiment Design

- **Curriculum Learning:**
  - Train a model sequentially on increasing modulus (e.g., mod-8, mod-16, mod-32, mod-64).
  - At each stage, transfer model weights to the next modulus (where each new task contains the previous as a special case).
  - Track LLC and accuracy/loss at each stage.

- **Finetuning:**
  - Train a model sequentially on a series of unrelated moduli (e.g., mod-5, mod-11, mod-23, mod-49, mod-53).
  - At each stage, transfer model weights to the next modulus, but the new task does **not** necessarily contain the previous one as a special case.
  - This tests how repeated finetuning on different but related tasks affects grokking and LLC, compared to curriculum learning.
  - Track LLC and accuracy/loss at each stage.

- **Direct Training:**
  - Train a separate model directly on the hardest modulus (e.g., mod-64 or mod-53).
  - Track LLC and accuracy/loss throughout training.

### Key Differences
- **Curriculum Learning**: Each new task is strictly harder and contains the previous as a subset, so the model can build on prior knowledge.
- **Finetuning**: Each new task is different and does not contain the previous one, so the model is repeatedly adapted to new, unrelated modular addition tasks.
- **Direct Training**: No transfer or staged learning; the model is trained from scratch on the hardest task.

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

It seems that both curriculum learning and finetuning can significantly decrease the number of steps to start grokking and  accelerate the rate of grokking, when applied to a final modulus task that a model is known to have the capacity to grokk on. However, grokking doesn't seem to ever take place for many numbers and can be altered when modifying seed and weight decay parameters in direct training, curriculum learning, or finetuning, and it is unclear whether either transfer approach improves or worsens adaptability (either through memorization or grokking): when a previous model grokked, the subsequent models did not always show grokking, but when they did they (often) did so much quicker compared to direct training.

- /results contains plots comparing curriculum, finetuning, and direct training in terms of accuracy, loss, and LLC.

## Files
- `main_experiment/curriculum_grokking_llc.ipynb`: Main experiment notebook
- `requirements.txt`: lists dependencies

## References
- [devinterp GitHub](https://github.com/timaeus-research/devinterp)
- Original grokking experiment: `devinterp/examples/grokking.ipynb`

## Contact
For questions or contributions, see the devinterp repository or open an issue.
