# Research Proposal: PID-Enhanced EEG/fNIRS Decoding

## 1. Background & Motivation
- **The Problem**: Current BCI/neuroimaging decoding often treats features as independent or uses "black box" deep learning. We lack understanding of *how* different signals contribute to the decoding.
- **The Opportunity**: EEG and fNIRS provide complementary information (Neurovascular Coupling).
    - **EEG**: High temporal resolution, measures electrical activity.
    - **fNIRS**: High spatial resolution (cortical), measures hemodynamic response (HbO, HbR).
- **The Hypothesis**: Successful decoding of complex cognitive states relies on the **synergistic** interaction between fast electrical dynamics (EEG) and slow metabolic support (fNIRS), not just their linear combination.

## 2. Research Objectives
1.  **Quantify Cross-Modal Information**: Use PID to decompose the information about a target stimulus ($Y$) provided by EEG features ($X_{eeg}$) and fNIRS features ($X_{fnirs}$).
    - How much is **Redundant**? (Do they tell us the same thing?)
    - How much is **Unique**? (What does fNIRS see that EEG misses, and vice versa?)
    - How much is **Synergistic**? (Does the *relationship* between EEG and fNIRS predict the state?)
2.  **PID-Guided Feature Selection**: Develop a feature selection algorithm that maximizes **Synergy** and **Uniqueness** while minimizing **Redundancy**.
    - Compare with standard methods (e.g., mRMR - min-Redundancy Max-Relevance).
3.  **Enhance Decoding Accuracy**: Test if PID-selected features improve classification accuracy in a BCI task (e.g., Motor Imagery or Mental Arithmetic).

## 3. Methodology
### A. Data Collection
- Simultaneous EEG (e.g., 32-64 channels) and fNIRS (e.g., prefrontal/motor cortex) recording.
- Task: Cognitive load task (n-back) or Motor Imagery.

### B. Feature Extraction
- **EEG**: Band power (Alpha, Beta, Theta), ERPs, Complexity measures.
- **fNIRS**: $\Delta [HbO]$, $\Delta [HbR]$, Slope, Mean.

### C. PID Implementation
- **Measure**: Use a continuous PID estimator (e.g., $I_{ccs}$ or Gaussian PID if appropriate) to handle continuous feature values.
- **Target ($Y$)**: Class labels (Rest vs. Task).
- **Sources ($X_1, X_2$)**: EEG feature vector, fNIRS feature vector.

### D. Validation
- **Metric**: Decoding Accuracy (SVM/LDA/CNN), F1-score.
- **Baseline**: Unimodal decoding (EEG only, fNIRS only) and standard concatenation fusion.

## 4. Expected Outcomes
- A quantitative map of "Information Flow" between electrical and hemodynamic domains.
- A novel "Synergy-based" fusion strategy that outperforms traditional concatenation.
- Biological insight: Does high synergy correlate with efficient neurovascular coupling?

## 5. Potential Challenges
- **Timescale Mismatch**: EEG is ms, fNIRS is seconds. How to align windows for PID? (Proposed solution: Feature-level fusion over trial windows).
- **Estimation Bias**: High-dimensional feature spaces. (Proposed solution: Pairwise PID or dimensionality reduction before PID).
