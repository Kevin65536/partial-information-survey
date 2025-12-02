# Migration Plan: From Theory to Implementation

## 1. Repository Summary

**Current Status**: Theoretical Framework & Literature Review
**Goal**: Develop a PID-guided multimodal pretraining framework for EEG and fNIRS.

### Key Outcomes
1.  **Theoretical Framework: Explicit Latent Partitioning (ELP)**
    *   **Core Idea**: Decompose the latent space into dedicated query tokens: $z_r$ (Redundancy), $z_{u\_eeg}$ (EEG Unique), $z_{u\_fnirs}$ (fNIRS Unique), and $z_s$ (Synergy).
    *   **Mechanism**: Use geometric constraints (Alignment, Orthogonality) as proxies for information-theoretic quantities to avoid complex MI estimation.
    *   **Training**: Mixed-batch training with "Routing via Masking" (Cross-modal, Uni-modal, Joint contexts).
    *   **Reference**: `script/pid_mcm_proposal.md`

2.  **Analysis Notes**
    *   Practical guidelines for generating synthetic data with known PID structure and verifying decomposition using linear baselines (CCA).
    *   **Reference**: `notes/pid_explicit_decomposition_analysis.md`

---

## 2. New Workspace Setup

**Recommended Name**: `pid-mcm-implementation`
**Directory Structure**:
```text
pid-mcm-implementation/
├── data/
│   ├── synthetic/          # Toy datasets for verification
│   └── raw/                # Real-world EEG-fNIRS datasets
├── src/
│   ├── models/             # ELPEncoder, Transformer backbone
│   ├── losses/             # Alignment, Orthogonality, Synergy losses
│   ├── data/               # Datasets, Masking logic
│   └── utils/              # Metrics (HSIC), Visualization
├── experiments/            # Configs and scripts for runs
├── notebooks/              # Exploratory analysis
└── docs/                   # Theory notes, API docs
```

---

## 3. Detailed Implementation Roadmap

### Phase 1: Theoretical Refinement & Synthetic Verification
*Goal: Validate the ELP logic on controlled data before tackling real-world noise.*

1.  **Formalize Loss Functions**:
    *   Implement `AlignmentLoss` (MSE between $z_r$ views).
    *   Implement `OrthogonalityLoss` (Cosine similarity penalty between $z_r, z_u, z_s$).
    *   Implement `SynergyLoss` (Penalty for *lack* of change in $z_s$ under masking).
2.  **Synthetic Data Generator**:
    *   Create a toy dataset where inputs $X_1, X_2$ are generated from independent latent sources $w_r, w_{u1}, w_{u2}, w_s$.
    *   Ensure ground truth is known: $R, U, S$ are mathematically defined.
    *   *Reference*: `notes/pid_explicit_decomposition_analysis.md`.
3.  **Proof of Concept**:
    *   Train a small ELP model on synthetic data.
    *   **Success Metric**: Can the model recover the correlation structure? (e.g., $z_r$ correlates with $w_r$, but not $w_u$).

### Phase 2: Real-World Data Selection & Pipeline
*Goal: Prepare the testbed for the actual research.*

1.  **Data Selection**:
    *   **Requirements**: Simultaneous EEG + fNIRS, reasonable sample size, cognitive tasks (to induce synergy).
    *   **Candidates**:
        *   **OpenBMI**: Motor imagery tasks (good for redundancy/unique).
        *   **Shin et al. (2018)**: Mental arithmetic (good for cognitive load/synergy).
        *   **BIP Datasets**: Standard BCI benchmarks.
2.  **Preprocessing Pipeline**:
    *   **EEG**: Bandpass filter (1-50Hz), downsample (e.g., 200Hz).
    *   **fNIRS**: Convert to HbO/HbR, bandpass (0.01-0.2Hz), resample to match EEG (or handle multi-rate in model).
    *   **Windowing**: Slice into synchronized epochs (e.g., 2-4 seconds).

### Phase 3: Model Implementation (ELP)
*Goal: Build the full architecture described in the proposal.*

1.  **Backbone**:
    *   Implement a Transformer Encoder that accepts tokenized EEG/fNIRS patches.
    *   Add **Learnable Query Tokens**: `[RED]`, `[UNI_E]`, `[UNI_F]`, `[SYN]`.
2.  **Masking Strategy**:
    *   Implement `MaskGenerator`:
        *   **Cross-Modal**: Mask EEG heavily, keep fNIRS (trains $z_r$).
        *   **Uni-Modal**: Mask fNIRS completely (trains $z_u$).
        *   **Joint**: Random masking (trains $z_s$).

### Phase 4: Evaluation & Comparison
*Goal: Demonstrate superiority over baselines.*

1.  **Baselines**:
    *   **Vanilla MAE**: Single reconstruction loss, no latent partitioning.
    *   **Contrastive Learning**: Standard CLIP-style alignment (captures only $R$).
    *   **CCA**: Linear baseline for subspace decomposition.
2.  **Metrics**:
    *   **Reconstruction Quality**: MSE/Correlation of reconstructed signals.
    *   **PID Quality**:
        *   **Disjointness**: $I(z_r; z_u) \approx 0$ (estimated via HSIC).
        *   **Synergy**: Does $z_s$ improve downstream classification over $z_r + z_u$?
    *   **Downstream Task**: Classification accuracy on Mental Arithmetic / Motor Imagery.

## 4. Immediate Next Steps
1.  Initialize the `pid-mcm-implementation` repository.
2.  Create the `SyntheticDataset` class to start Phase 1.
3.  Draft the `ELPEncoder` PyTorch module.
