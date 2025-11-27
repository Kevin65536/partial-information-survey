import numpy as np
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

def calculate_mi(x, y):
    """
    Calculate Mutual Information between x and y.
    x: shape (n_samples, n_features_x) or (n_samples,)
    y: shape (n_samples,)
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return mutual_info_regression(x, y, random_state=42)[0]

def simulate_identical_variables_with_noise(n_samples=10000, noise_level=1.0):
    print(f"\n--- Simulation 1: Identical Variables with Independent Noise (Noise Level={noise_level}) ---")
    
    # Source signal S
    S = np.random.normal(0, 1, n_samples)
    
    # Two variables that are copies of S with different independent noise
    # X1 = S + N1
    # X2 = S + N2
    N1 = np.random.normal(0, noise_level, n_samples)
    N2 = np.random.normal(0, noise_level, n_samples)
    
    X1 = S + N1
    X2 = S + N2
    
    # Calculate MI terms
    # I(S; X1)
    i_s_x1 = calculate_mi(X1, S)
    # I(S; X2)
    i_s_x2 = calculate_mi(X2, S)
    # I(S; [X1, X2])
    X_joint = np.column_stack((X1, X2))
    i_s_x1x2 = calculate_mi(X_joint, S)
    
    print(f"I(S; X1) = {i_s_x1:.4f}")
    print(f"I(S; X2) = {i_s_x2:.4f}")
    print(f"I(S; X1, X2) = {i_s_x1x2:.4f}")
    
    # PID Decomposition (MMI Definition)
    # Redundancy = min(I(S; X1), I(S; X2))
    redundancy = min(i_s_x1, i_s_x2)
    
    # Unique Information
    unique_1 = i_s_x1 - redundancy
    unique_2 = i_s_x2 - redundancy
    
    # Synergy
    # I(S; X1, X2) = Redundancy + Unique1 + Unique2 + Synergy
    synergy = i_s_x1x2 - (redundancy + unique_1 + unique_2)
    
    print("\nPID Results (MMI Definition):")
    print(f"Redundancy: {redundancy:.4f} ({redundancy/i_s_x1x2*100:.1f}%)")
    print(f"Unique X1:  {unique_1:.4f} ({unique_1/i_s_x1x2*100:.1f}%)")
    print(f"Unique X2:  {unique_2:.4f} ({unique_2/i_s_x1x2*100:.1f}%)")
    print(f"Synergy:    {synergy:.4f} ({synergy/i_s_x1x2*100:.1f}%)")
    
    return redundancy, unique_1, unique_2, synergy

def simulate_multichannel_high_noise(n_samples=10000, n_channels=3, noise_level=10.0):
    print(f"\n--- Simulation 2: Multi-channel Signal with High Noise (Channels={n_channels}, Noise Level={noise_level}) ---")
    
    # Source signal S (e.g., underlying brain state)
    S = np.random.normal(0, 1, n_samples)
    
    # Multiple channels recording S with high independent noise
    # Xi = S + Ni
    X = []
    for i in range(n_channels):
        Ni = np.random.normal(0, noise_level, n_samples)
        X.append(S + Ni)
    
    X = np.column_stack(X)
    
    # Calculate Pairwise MI with S for each channel
    mi_singles = []
    for i in range(n_channels):
        mi = calculate_mi(X[:, i], S)
        mi_singles.append(mi)
        print(f"I(S; X{i+1}) = {mi:.4f}")
        
    # Calculate Joint MI
    i_s_joint = calculate_mi(X, S)
    print(f"I(S; X_joint) = {i_s_joint:.4f}")
    
    # PID Analysis (Simplified for Multi-variable)
    # Redundancy (MMI) = min(I(S; Xi))
    redundancy = min(mi_singles)
    
    # Total Unique (Sum of Uniques) + Synergy
    # Interaction Information = I(S; X_joint) - sum(I(S; Xi)) (This is Interaction Info, not exactly Synergy)
    
    # Let's look at Synergy in the context of MMI
    # Synergy = I(S; X_joint) - max(I(S; Xi)) (This is a lower bound on Synergy if we ignore Unique? No, MMI is simpler)
    # In MMI:
    # I(S; X1...Xn) = Redundancy + Sum(Unique) + Synergy terms... it gets complex for n > 2.
    # But we can look at "Synergistic Gain": I(S; X_joint) - max(I(S; Xi))
    # If I(S; X_joint) >> max(I(S; Xi)), it implies Synergy.
    
    max_single_mi = max(mi_singles)
    gain = i_s_joint - max_single_mi
    
    print("\nAnalysis:")
    print(f"Redundancy (MMI): {redundancy:.4f}")
    print(f"Max Single MI:    {max_single_mi:.4f}")
    print(f"Joint MI:         {i_s_joint:.4f}")
    print(f"Gain (Joint - Max): {gain:.4f}")
    print(f"Gain Ratio:       {gain/max_single_mi:.2f}x")
    
    return redundancy, i_s_joint

if __name__ == "__main__":
    # Scenario 1: Identical variables with noise
    # Low noise
    simulate_identical_variables_with_noise(noise_level=0.5)
    # High noise
    simulate_identical_variables_with_noise(noise_level=2.0)
    
    # Scenario 2: High noise EEG-like signal
    # simulate_multichannel_high_noise(n_channels=2, noise_level=5.0)
    # simulate_multichannel_high_noise(n_channels=4, noise_level=5.0)
    # simulate_multichannel_high_noise(n_channels=8, noise_level=5.0)
