# Grep Directory — notebook.ipynb

> Cell-by-cell index of the notebook's technical content.
> 83 cells total (52 markdown, 31 code). Follows Chollet's 8-step ML workflow.

---

to convert to pdf and html run
```jupyter nbconvert --to webpdf --no-input notebook.ipynb; jupyter nbconvert --to html --no-input notebook.ipynb```

## Document Map

| Step | Cells | Title |
|------|-------|-------|
| — | 0 | Title page |
| 1 | 1-3 | Define the problem |
| 2 | 4-9 | Identify and understand the data |
| 3 | 10-15 | Choose an evaluation protocol |
| 4 | 16-31 | Prepare the data (loading, EDA, splitting, scaling) |
| 5 | 32-54 | Establish a baseline and pick a starting model |
| 6 | 55-64 | Develop the model: architecture deep-dive |
| 7 | 65-74 | Model improvement and threshold tuning |
| 8 | 75-80 | Final evaluation and deployment considerations |
| — | 81 | Glossary |
| — | 82 | Bibliography |

---

## Cell-by-Cell Index

### Header (cell 0)

| Field | Value |
|-------|-------|
| Module | CM3015 Machine Learning and Neural Networks |
| Task | Credit Card Fraud Detection with a Feedforward MLP |
| Student | cy150 |
| Workflow | Chollet: problem → data → evaluation → prep → baseline → model → tuning → final eval |

---

### Step 1 — Define the problem (cells 1-3)

| Cell | Heading | Content |
|------|---------|---------|
| 1 | `## Step 1 — Define the problem` | Chollet workflow overview. Defines credit card fraud. |
| 2 | `### Problem Statement` | Financial losses from fraud, regulatory penalties, KYC overhead. |
| 3 | `### Success Metrics` → `#### Objective` | Primary objective: detect fraud, minimise false positives. Focus on minority-class performance. |

**Key terms introduced:** credit card fraud, chargebacks, false declines, KYC.

---

### Step 2 — Identify and understand the data (cells 4-9)

| Cell | Heading | Content |
|------|---------|---------|
| 4 | `## Step 2` | Dataset overview: 284,807 transactions, 492 frauds (~0.172%). European cardholders, 2-day window. Forward-reference to EDA in Step 4. |
| 5 | `### Nature of the Dataset` | 28 PCA features (V1-V28) + `Time` + `Amount` + `Class`. Example rows shown. |
| 6 | `### Dataset Licensing` | Database Contents License (DbCL). Author: Machine Learning Group — ULB. |
| 7 | `### Dataset Source` | Kaggle: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data` |
| 8 | `### Justification for this dataset` | Clear labels, numeric features, real-world class imbalance. |
| 9 | `#### Rationale` / `#### Limitations` | Matches problem framing, data modality, evaluation setup, experimental discipline. Limitations: anonymised PCA (no feature interpretation), potential time effects. Mitigations stated. |

**Key numbers:** 284,807 rows, 492 frauds, 30 features, 0.172% fraud rate, 2-day window.

---

### Step 3 — Choose an evaluation protocol (cells 10-15)

| Cell | Heading | Content |
|------|---------|---------|
| 10 | `## Step 3` / `### Holdout Protocol` | 4-point protocol: train/val/test split; fit transforms on train only; val for selection only; test used once. Time-aware split rationale. |
| 11 | `### Evaluation metrics` | Introductory sentence linking to pre-defined success criteria. |
| 12 | `### Primary Metrics` | Table: **PR AUC**, **Recall**, **F1 Score** — each with "what it measures" and "why it matters". |
| 13 | `### Secondary Metrics` | Table: **Confusion matrix at threshold**, **Error rates** (FNR, FPR), **Calibration check** (binning). |
| 14 | `#### Justification for evaluation metrics` | Single metric misleading for rare events; primary for comparison, secondary for context. |
| 15 | `### Implications of the evaluation metrics` | Precision → workload/friction; Recall → loss prevention; PR AUC → ranking quality; CM → operating-point tradeoffs. |

**Primary metrics:** PR AUC, Recall, F1.
**Secondary metrics:** Confusion matrix, FNR/FPR, calibration.
**Rejected metric:** Accuracy (misleading under imbalance).

---

### Step 4 — Prepare the data (cells 16-31)

| Cell | Type | Heading | Content |
|------|------|---------|---------|
| 16 | md | `## Step 4` / `### Data preparation plan` | Section heading. |
| 17 | md | *(body text)* | 8-point plan: time-aware split, fit on train only, preserve imbalance, track drift, threshold sweep, calibrate, retrain for drift. |
| 18 | md | `### Data checks and class imbalance` | Checklist: missing values, outliers, scaling, class weights/resampling, reproducibility notes. |
| 19 | md | `### Step 4 implementation (code)` | Description: loads from `data/creditcard.csv`, fallback to kagglehub. Expected columns. |
| 20 | **code** | *(pip install + download)* | `%pip install kagglehub`, downloads dataset, prints shape and head. |
| 21 | **code** | *(data loading — main)* | Loads CSV from `data/creditcard.csv`, or fetches via kagglehub. Prints shape, head, fraud rate. |
| 22 | md | `### EDA — Visualizing the Dataset` | Table mapping 6 EDA focuses to rationale. |
| 23 | **code** | *(EDA 1)* | Class distribution: bar chart + pie chart. Prints exact counts and percentages. |
| 24 | **code** | *(EDA 2)* | Transaction amount distributions: histograms (raw + log), box plots, fraud vs non-fraud overlays. |
| 25 | **code** | *(EDA 3)* | Transaction time: volume over time, fraud vs non-fraud density, fraud rate over time, cumulative fraud count. |
| 26 | **code** | *(EDA 4)* | Correlation heatmap: per-feature correlation with `Class`, top-15 correlation matrix. |
| 27 | **code** | *(EDA 5)* | Violin + strip plots for 8 most discriminative features by class. |
| 28 | **code** | *(EDA 6)* | t-SNE 2D projection (all fraud + 3,000 sampled non-fraud). `perplexity=30`, `max_iter=1000`. |
| 29 | md | `### EDA Summary` | 6-point summary: extreme imbalance, amount differences, temporal patterns, feature correlations (V14, V17, V12, V10), violin separation, t-SNE partial separability. |
| 30 | **code** | *(splitting + scaling)* | Time-aware sort → 70/15/15 split → `StandardScaler` fit on train only → `class_weight` computed as `neg/pos`. |
| 31 | md | `### Split Success` | Confirms 3 non-overlapping subsets, 30 features, sensible fraud rates, scaling on train only. |

**Key variables created:**
- `df` — full DataFrame (284,807 × 31)
- `X_train`, `X_val`, `X_test` — raw feature arrays
- `y_train`, `y_val`, `y_test` — label arrays
- `X_train_scaled`, `X_val_scaled`, `X_test_scaled` — StandardScaler-transformed
- `scaler` — fitted `StandardScaler` instance
- `class_weight` — `{0: 1.0, 1: neg/pos}` (~578×)
- `feature_names` — list of 30 column names

**Split ratios:** 70% train / 15% val / 15% test (time-ordered).

**Key EDA findings:** V14, V17, V12, V10 most correlated with fraud. PCA components uncorrelated with each other (as expected). t-SNE shows partial separability.

---

### Step 5 — Establish a baseline and pick a starting model (cells 32-54)

| Cell | Type | Heading | Content |
|------|------|---------|---------|
| 32 | md | `## Step 5` | Intro: why baselines matter, starting model rationale. |
| 33 | md | `### 5.0 — Pre-modelling data sanity check` | Catches NaNs, Infs, duplicates, unexpected ranges after splitting. |
| 34 | **code** | *(sanity check)* | Prints NaN/Inf counts, duplicate count, feature range stats, label counts per split. |
| 35 | md | `### 5.0.1 — Post-split visual sanity checks` | Visual verification: class distributions, feature distributions across splits, Amount/Time consistency. |
| 36 | **code** | *(visual checks)* | 2×3 subplot: split sizes bar chart, fraud rate per split, class counts stacked bar, Amount/Time/V14 distribution overlays across splits. |
| 37 | md | `### 5.1 — Trivial baseline` | Always predict class 0. Expected: Recall=0, Precision=0, PR AUC=fraud prevalence. |
| 38 | **code** | *(trivial baseline)* | `y_val_pred_trivial = np.zeros_like(y_val)`. Computes precision, recall, F1, PR AUC, confusion matrix. |
| 39 | md | `### 5.2 — Logistic regression baseline` | `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42`. |
| 40 | **code** | *(logistic regression)* | Fits `LogisticRegression`, predicts probabilities, computes same metrics, prints confusion matrix. |
| 41 | md | `### 5.3 — Starting model: feedforward MLP` | Architecture: `Input(30) → Dense(128,ReLU) → Dropout(0.4) → Dense(64,ReLU) → Dropout(0.3) → Dense(32,ReLU) → Dropout(0.3) → Dense(1,sigmoid)`. 14,337 parameters. |
| 42 | **code** | *(MLP definition)* | Builds Keras Sequential MLP. Seeds set: `SEED=42`. Compiles with `adam`, `binary_crossentropy`, `AUC` metric. Prints `model.summary()`. |
| 43 | md | `### 5.4 — Train the starting MLP` | Explains training loop: forward pass, loss, class weighting, backprop, early stopping. |
| 44 | **code** | *(MLP training)* | `EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)`. `epochs=100`, `batch_size=2048`. Uses `class_weight`. |
| 45 | **code** | *(training curves)* | Plots loss and AUC curves (train vs val) over epochs. Auto-detects AUC key name across TF versions. |
| 46 | md | `### 5.5 — Evaluate MLP + compare` | Threshold=0.5. Fair comparison of all three models. |
| 47 | **code** | *(MLP val metrics)* | `mlp.predict()` → threshold 0.5 → Recall, Precision, F1, PR AUC, confusion matrix. |
| 48 | **code** | *(comparison table)* | DataFrame: Trivial / LogReg / MLP side-by-side. Styled with `highlight_max`. |
| 49 | **code** | *(PR curve overlay)* | All three PR curves on one plot. |
| 50 | md | `### 5.6 — Confusion matrix heatmaps` | Side-by-side seaborn heatmaps. |
| 51 | **code** | *(CM heatmaps)* | 1×3 subplot: Trivial, LogReg, MLP confusion matrices. |
| 52 | md | `### 5.7 — Predicted score distributions` | Histograms of P(fraud) by true label. Separation = discrimination quality. |
| 53 | **code** | *(score distributions)* | LogReg + MLP probability histograms, fraud vs non-fraud, threshold line at 0.5. |
| 54 | md | `### Step 5 — Summary` | 6-point interpretation + "what comes next" (Steps 6, 7, 8). |

**Key variables created:**
- `y_val_pred_trivial`, `y_val_prob_trivial` — trivial baseline outputs
- `lr` — fitted `LogisticRegression`
- `y_val_pred_lr`, `y_val_prob_lr` — logistic regression outputs
- `pr_auc_trivial`, `pr_auc_lr`, `pr_auc_mlp` — PR AUC scores
- `mlp` — trained Keras `Sequential` model
- `y_val_prob_mlp`, `y_val_pred_mlp` — MLP outputs at threshold 0.5
- `history` — Keras training history object

**Hyperparameters (starting MLP):**
- Layers: 128 → 64 → 32 → 1
- Dropout: 0.4, 0.3, 0.3
- Optimizer: Adam (default lr=1e-3)
- Loss: binary_crossentropy
- Batch size: 2048
- Epochs: 100 (early stopping, patience=10)
- Seed: 42

---

### Step 6 — Develop the model: architecture deep-dive (cells 55-64)

| Cell | Heading | Content |
|------|---------|---------|
| 55 | `## Step 6` / `### 6.1 — Feedforward neural networks` | FNN definition. Layer equation: h^(l) = σ(W^(l) h^(l-1) + b^(l)). Universal approximation theorem (Hornik 1989). Why feedforward for tabular data (fixed-size, no temporal dependency, efficient, interpretable capacity). |
| 56 | `### 6.2 — The MLP` | Dense layer definition. Full forward-pass equations (h1→h2→h3→ŷ). Parameter count table (14,337 total). Funnel/bottleneck design rationale (128→64→32). Parameter-to-sample ratio ~1:14. |
| 57 | `### 6.3 — Justification: MLP over other architectures` | Comparison table: MLP vs CNN vs RNN/LSTM vs Transformer vs Tree-based. CNN: no spatial locality. RNN: not sequential per-row. Transformer: overkill for 30-element vector. Trees: competitive but outside assignment scope. |
| 58 | `### 6.4 — Activation functions` | **ReLU** (hidden): max(0,z). Advantages: computational efficiency, sparse activation, mitigates vanishing gradient. Risk: dying ReLU. **Sigmoid** (output): 1/(1+e^-z). Maps to (0,1) for probability. Table: why not sigmoid/tanh in hidden layers (vanishing gradients, saturated outputs, slower convergence). |
| 59 | `### 6.5 — Loss function and optimisation` | **Binary cross-entropy**: L = -[y log ŷ + (1-y) log(1-ŷ)]. Penalises confident wrong predictions. Combined with class weights → weighted BCE. **Adam** optimiser: combines momentum + adaptive per-parameter learning rates. Why Adam over SGD/RMSprop (stable convergence, less LR tuning). |
| 60 | `### 6.6 — Regularisation strategy` | **Dropout**: p=0.4 (layer 1), p=0.3 (layers 2-3). Trains ensemble of sub-networks. **Early stopping**: patience=10, restore_best_weights. Prevents overfitting. Why not L2 weight decay (Dropout sufficient, interactions harder to disentangle). |
| 61 | `### 6.7 — Architecture summary and layer-by-layer analysis` | Weight distribution analysis intro. Healthy weights: centred near zero, moderate spread. Dead neurons: collapsed to zero. |
| 62 | **code** | Iterates over Dense layers, prints weight/bias shapes, mean, std, min, max. Plots weight distribution histograms per layer. |
| 63 | md | `#### Interpreting the weight analysis` | Table of layer stats (dense_4 through dense_7): weight shapes, means near 0, stds 0.13-0.27, ranges [-0.85, +0.66]. All healthy, no dead neurons, output layer slightly wider distribution. |
| 64 | md | `### 6.8 — Step 6 summary` | Table: feedforward networks, MLP structure, architecture justification, ReLU/sigmoid, BCE+Adam, Dropout+early stopping, weight analysis — all with key takeaways. |

**Mathematical formulas defined:**
- FNN layer equation
- Full MLP forward pass (4 equations)
- ReLU
- Sigmoid
- Binary cross-entropy
- Adam (conceptual)

---

### Step 7 — Model improvement and threshold tuning (cells 65-74)

| Cell | Type | Heading | Content |
|------|------|---------|---------|
| 65 | md | `## Step 7` | Three-stage plan: threshold optimisation, hyperparameter experimentation, multi-seed stability. |
| 66 | md | `### 7.1 — Decision threshold optimisation` | Default 0.5 rarely optimal for imbalanced problems. F1 = 2·(P·R)/(P+R). Sweep to maximise F1. |
| 67 | **code** | *(threshold sweep)* | Sweeps `np.arange(0.01, 1.00, 0.01)`. Plots Precision/Recall/F1 vs threshold. Prints best threshold and metrics. Confusion matrix heatmap at best threshold. |
| 68 | md | `### 7.2 — Hyperparameter experimentation` | 3 axes: layer widths ([64,32,16] / [128,64,32] / [256,128,64]), dropout rates ([0.3,0.2,0.2] / [0.4,0.3,0.3] / [0.5,0.4,0.4]), learning rate (1e-4 / 1e-3 / 5e-3). 7 configs total. |
| 69 | **code** | *(hyperparameter experiments)* | `build_mlp_flex()` function. `train_and_evaluate()` function. Loops over 7 configs, trains each, records PR AUC + F1. Bar chart comparison. Selects best config by PR AUC. |
| 70 | md | `### 7.3 — Multi-seed stability check` | 5 seeds: [42, 123, 456, 789, 2024]. Reports mean ± std. Target: std < 0.02. |
| 71 | **code** | *(stability check)* | Retrains best config with each seed. Prints per-seed PR AUC and F1. Computes mean ± std. |
| 72 | md | `### 7.4 — Final validation evaluation` | Definitive validation eval with best config + best threshold. Classification report, confusion matrices, PR curve comparison. |
| 73 | **code** | *(final val evaluation)* | Classification report at tuned threshold. Side-by-side CM heatmaps (starting MLP vs tuned). PR curve overlay (all models including tuned). |
| 74 | md | `### 7.5 — Step 7 summary` | Table: threshold optimisation → F1-maximising threshold, hyperparameter experiments → best config, multi-seed → reproducibility confirmed. |

**Key variables created:**
- `best_threshold` / `final_threshold` — F1-maximising threshold
- `best_config_result` — best hyperparameter config by PR AUC
- `final_model` — retrained model with best config
- `final_y_prob` — final model's validation probabilities
- `stability_results` — list of per-seed results
- `configs` — list of 7 hyperparameter config dicts

**Hyperparameter search space:**
- Layer widths: [64,32,16], [128,64,32], [256,128,64]
- Dropout rates: [0.3,0.2,0.2], [0.4,0.3,0.3], [0.5,0.4,0.4]
- Learning rates: 1e-4, 1e-3, 5e-3
- Seeds tested: 42, 123, 456, 789, 2024

---

### Step 8 — Final evaluation and deployment considerations (cells 75-80)

| Cell | Type | Heading | Content |
|------|------|---------|---------|
| 75 | md | `## Step 8` / `### 8.1 — Test set evaluation` | Test set held out since beginning. Never used for training, threshold tuning, or HP selection. Applies final model + optimised threshold. |
| 76 | **code** | *(test evaluation)* | `final_model.predict(X_test_scaled)` → threshold → PR AUC, F1, Recall, Precision, confusion matrix, classification report. CM heatmap. |
| 77 | md | `### 8.2 — Validation vs test comparison` | Similar performance = healthy. Large gap = overfitting to val. |
| 78 | **code** | *(val vs test table)* | Computes val and test metrics. Prints side-by-side table with delta (Test−Val). |
| 79 | md | `### 8.3 — Deployment considerations` | Real-time scoring pipeline, risk tiers, monitoring/retraining (concept drift), model governance, limitations (anonymised features, 2-day window, single model family). |
| 80 | md | `### 8.4 — Conclusion` | Summary table of all 8 steps. 5 key takeaways: threshold tuning matters most, MLP captures non-linear patterns, multi-seed stability confirms reproducibility, strict train/val/test discipline, limitations acknowledged. |

**Key variables created:**
- `y_test_prob`, `y_test_pred` — test set predictions
- `pr_auc_test`, `f1_test` — test set metrics

---

### Appendix (cells 81-82)

| Cell | Heading | Content |
|------|---------|---------|
| 81 | `## Glossary of terms` | 35 terms defined: binary classification, class imbalance, PCA, train/val/test sets, data leakage, standardization, Dense layer, Dropout, ReLU, sigmoid, logits, decision threshold, confusion matrix, TP/FP/TN/FN, precision, recall, F1, ROC, AUC, PR AUC, overfitting, regularization, hyperparameter, learning rate, optimizer, loss function, early stopping, calibration, concept drift, baseline model. |
| 82 | `### Bibliography & Citations` | 10 references. Primary authors: Carcillo, Dal Pozzolo, Le Borgne, Bontempi (ULB group). Covers streaming fraud detection, Spark framework, unsupervised+supervised combination, realistic modelling, domain adaptation, incremental learning, and the practitioner handbook. |

---

## Key Technical Decisions (cross-reference)

| Decision | Where defined | Where implemented |
|----------|--------------|-------------------|
| PR AUC as primary metric | Cell 12 (Step 3) | Cells 38, 40, 47, 67, 69, 73, 76 |
| Time-aware split (70/15/15) | Cell 10 (Step 3), Cell 17 (Step 4) | Cell 30 |
| StandardScaler fit on train only | Cell 10 (Step 3), Cell 17 (Step 4) | Cell 30 |
| Class weighting (neg/pos ratio) | Cell 18 (Step 4) | Cell 30 (computed), Cell 44 (used in training) |
| MLP architecture (128→64→32) | Cell 41 (Step 5) | Cell 42 |
| Dropout rates (0.4, 0.3, 0.3) | Cell 41 (Step 5), Cell 60 (Step 6) | Cell 42 |
| Early stopping (patience=10) | Cell 43 (Step 5), Cell 60 (Step 6) | Cell 44 |
| Default threshold 0.5 | Cell 46 (Step 5) | Cell 47 |
| Threshold optimisation (F1-max) | Cell 66 (Step 7) | Cell 67 |
| Hyperparameter search (7 configs) | Cell 68 (Step 7) | Cell 69 |
| Multi-seed stability (5 seeds) | Cell 70 (Step 7) | Cell 71 |
| Test set used exactly once | Cell 10 (Step 3), Cell 75 (Step 8) | Cell 76 |

---

## Libraries Used

| Library | Purpose | First used |
|---------|---------|------------|
| `numpy` | Array operations | Cell 21 |
| `pandas` | DataFrame operations | Cell 21 |
| `matplotlib` | Plotting | Cell 23 |
| `seaborn` | Heatmaps, violin plots | Cell 23 |
| `sklearn.preprocessing.StandardScaler` | Feature scaling | Cell 30 |
| `sklearn.metrics` | PR AUC, F1, precision, recall, confusion matrix | Cell 38 |
| `sklearn.linear_model.LogisticRegression` | Baseline classifier | Cell 40 |
| `sklearn.manifold.TSNE` | Dimensionality reduction for EDA | Cell 28 |
| `tensorflow` / `keras` | MLP definition, training, inference | Cell 42 |
| `kagglehub` | Dataset download fallback | Cell 20 |

---

## Data Flow Summary

```
Raw CSV (284,807 × 31)
  │
  ├─ Sort by Time
  │
  ├─ 70% Train (199,364)
  ├─ 15% Val   (42,721)
  └─ 15% Test  (42,722)
       │
       ├─ StandardScaler (fit on train)
       │
       ├─ Step 5: Trivial baseline → LogReg → Starting MLP (threshold 0.5)
       │
       ├─ Step 7: Threshold sweep → HP search (7 configs) → Multi-seed (5 seeds)
       │          → Final model + optimal threshold
       │
       └─ Step 8: Test set evaluation (one-time, unbiased)
```
