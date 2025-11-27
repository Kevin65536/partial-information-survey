# todo

1. test methods to extract the information atom amount or representation from EEG and fNIRS data. 

    - using method PIDF to quantize the information atom amount. 
    - reference to pid_explicit_decomposition_analysis.md, using methods like cca trying to find the representation of information atoms.   

2. go through past researchs combining pid with neuroscience

    ```
    A strong convergence between PID and Tononi’s $\Phi$ measure of consciousness. This has led to $\Phi$ ID (Integrated Information Decomposition), which uses PID to mathematically formalize the emergence of consciousness and causal decoupling in the brain
    ```
    ```
    Functional Connectivity:** Replacing standard pairwise correlation networks with high-order interaction networks (hypergraphs) to map brain dynamics.
    ```

3. investment in pretraining with pid

    - check if any research have employed pid in pretraining
    - check multimodal pretrain paradigms
    - rethink if it is workable to employ pid in multimodal pretraining, constrain the process of two modalities predicting each other


ps: some other good points

- "Instantaneous PID" metrics that can track how a relationship shifts from redundant to synergistic *in real-time* during a phase transition, without requiring a sliding window that blurs the dynamics.
- Understanding the "In-context Learning" of LLMs through the lens of PID. Do attention heads act redundantly (voting) or synergistically (assembling features)?
- *Faes et al. (2022, 2024)* and *Pinzuti et al. (2020)* have developed frameworks to calculate PID terms in the frequency domain (e.g., **Spectral O-Information**). This allows you to say, "Alpha band drives redundancy, while Gamma band drives synergy."
- *   **Stress Responses:** *Krohova (2019)* and *Pinto (2023)* showed that physiological stress (tilt test, mental arithmetic) shifts the system from redundant control to synergistic control. For example, paced breathing increases redundancy (fault tolerance) in cardiorespiratory control.
- Does the prefrontal cortex (PFC) switch from redundant to synergistic coding as cognitive load increases? Current fMRI papers suggest yes, but fNIRS allows you to test this in naturalistic settings (walking, moving) where fMRI fails.
- Is the information transferred from the electrical domain to the hemodynamic domain purely unique, or is there a synergistic interaction between frequency bands that predicts the hemodynamic response? This could define a new "Information-Theoretic Neurovascular Coupling" metric.

