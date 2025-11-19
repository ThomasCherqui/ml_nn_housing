## Neural Networks 70050‑092 – Coursework Summary

This repository is the artefact we hand in for the *Introduction to Machine Learning* assignment. It doubles as a concise report: each section states **what we implemented** and **why we made that choice**.

---

### Contents

| File | What we did |
| --- | --- |
| `part1_nn_lib.py` | Built a full MLP training stack with numpy only (Xavier init, linear layers, ReLU/Sigmoid/Identity, cross-entropy & MSE losses, manual backprop/trainer, min–max `Preprocessor`). |
| `part2_house_value_regression.py` | Implemented a PyTorch `Regressor` for `housing.csv`, including preprocessing, a small MLP, MSE-based training, lightweight hyperparameter search, and model pickling. |
| `part2_model.pickle` | The best model we saved (using the best hyperparameters) and trained on 80% of the dataset. |
| `housing.csv` / `iris.dat` | Datasets used in the write-up (California Housing + Iris). |
| `requirements.txt` | Exact dependency versions used while running the experiments. |

---

### Part 1 – Hand-crafted neural network (Iris)

- **Architecture**: `4 → 16 → 3` with ReLU then identity. We wanted something shallow yet expressive enough to hit high accuracy without obscuring the maths.
- **Initialisation**: Xavier for all affine layers to keep gradients stable; explained in the written answer about vanishing gradients.
- **Loss/metric**: We trained with categorical cross entropy (implemented as `CrossEntropyLossLayer`) and reported accuracy because the task is multi-class classification.
- **Training loop**: Custom `Trainer` uses SGD, minibatching, optional shuffling, and explicit gradient updates—every tensor operation is derived manually.
- **Preprocessing**: The `Preprocessor` stores per-feature min/max to normalise inputs and later invert them; we highlight in the report that this keeps each feature in `[0,1]`.

Running `python part1_nn_lib.py` executes the exact experiment described in the submission (train/val split on Iris, final accuracy printout).

---

### Part 2 – California house-value regressor

- **Preprocessing choices**:
  - Median imputation for numeric features while fitting, zeros at inference (prevents data leakage in cross-validation).
  - One-hot encoding of `ocean_proximity` with an explicit “None/unknown” bucket, so the model can handle unseen categories.
  - StandardScaler applied separately to `X` and `y`; this yielded faster convergence and lets us rescale predictions back to dollars.
- **Model**: MLP `13 → 64 → 32 → 1` with ReLU activations. This size was the best compromise between capacity and overfitting on our validation split.
- **Metric choice**: We explicitly chose **Mean Squared Error** as both training loss (`nn.MSELoss`) and evaluation metric because the coursework rubric emphasised a regression score. We also report RMSE (square root of MSE) for interpretability.
- **Optimisation**: We grid-searched learning rate, batch size, and epochs (see `perform_hyperparameter_search`) and selected the configuration with the lowest validation MSE.



