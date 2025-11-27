"""
PID Analysis for EEG/fNIRS Multimodal Data

This module implements Partial Information Decomposition (PID) analysis using the PIDF 
(Partial Information Decomposition for Feature Selection) method to analyze the information 
contributions of EEG and fNIRS signals for predicting cognitive events.

Key concepts:
- Redundancy: Information shared by multiple features about the target
- Unique: Information only one feature provides about the target  
- Synergy: Information only available when features are combined

Reference: PIDF paper methodology
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))
from load_simultaneous_data import SimultaneousData

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class PIDFResult:
    """Store detailed PIDF analysis results for a single feature."""
    feature_index: int
    feature_name: str
    mutual_information: float  # I(Xi; Y)
    synergy: float  # Synergistic contribution
    redundancy: float  # Redundant contribution
    unique_info: float  # Unique information = MI - Redundancy
    total_contribution: float  # MI + Synergy
    is_selected: bool
    redundant_with: List[int]  # Features this is redundant with
    synergistic_with: List[int]  # Features this has synergy with


class EnhancedPIDF:
    """
    Enhanced PIDF implementation with detailed PID decomposition results.
    
    This class extends the original PIDF algorithm to provide:
    - Detailed per-feature PID metrics
    - Confidence intervals for all estimates
    - Better error handling
    - Visualization support
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 feature_names: Optional[List[str]] = None,
                 num_iterations: int = 200, ci: float = 0.95):
        """
        Initialize the Enhanced PIDF analyzer.
        
        Parameters:
        - features: np.ndarray, shape (n_samples, n_features)
        - targets: np.ndarray, shape (n_samples,) or (n_samples, 1)
        - feature_names: Optional list of feature names
        - num_iterations: Training iterations for MLP
        - ci: Confidence interval level
        """
        self.num_iterations = num_iterations
        self.ci = ci
        
        # Ensure proper shapes
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
            
        self.n_features = features.shape[1]
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(self.n_features)]
        
        # Normalize and convert to torch
        self.features_np = features.round(decimals=2)
        self.targets_np = targets.round(decimals=2)
        
        self.features = torch.tensor(self.features_np, dtype=torch.float32, device=device)
        self.targets = torch.tensor(self.targets_np, dtype=torch.float32, device=device)
        
        self.features = self._normalize_torch(self.features)
        self.targets = self._normalize_torch(self.targets)
        
        # Results storage
        self.results: List[PIDFResult] = []
        self.feature_mi_matrix: Optional[np.ndarray] = None
        
    def _normalize_torch(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to zero mean and unit variance."""
        epsilon = 1e-8
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return (data - mean) / (std + epsilon)
    
    def _create_mlp(self, input_size: int, output_size: int) -> nn.Sequential:
        """Create an MLP regressor."""
        return nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, output_size)
        ).to(device)
    
    def _fit_mlp(self, features: torch.Tensor, targets: torch.Tensor, 
                 mlp: nn.Sequential, feature_mask: Optional[torch.Tensor] = None) -> float:
        """
        Train MLP and return final loss.
        
        The loss serves as a proxy for -MI (lower loss = higher predictability = higher MI)
        """
        if feature_mask is not None:
            features = features * feature_mask
            
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp.parameters(), lr=0.001)
        
        for _ in range(self.num_iterations):
            predictions = mlp(features)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return loss.item()
    
    def _calculate_confidence_interval(self, values: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Calculate mean and confidence interval for a set of values."""
        mean = np.mean(values)
        if len(values) < 2 or np.std(values) == 0:
            return mean, (mean, mean)
        
        sem = stats.sem(values)
        ci = stats.t.interval(self.ci, len(values)-1, loc=mean, scale=sem)
        ci = tuple(mean if np.isnan(val) else val for val in ci)
        return mean, ci
    
    def estimate_mi(self, feature_indices: List[int]) -> np.ndarray:
        """
        Estimate mutual information I(X_indices; Y) using MLP prediction loss.
        
        Returns array of loss differences across multiple runs.
        """
        n_runs = 3
        
        # Mask for no features
        zero_mask = torch.zeros(self.n_features, device=device)
        
        # Mask for selected features
        feature_mask = torch.zeros(self.n_features, device=device)
        for idx in feature_indices:
            feature_mask[idx] = 1
        
        losses_without = []
        losses_with = []
        
        for _ in range(n_runs):
            mlp = self._create_mlp(self.n_features, self.targets.shape[-1])
            losses_without.append(self._fit_mlp(self.features, self.targets, mlp, zero_mask))
            
            mlp = self._create_mlp(self.n_features, self.targets.shape[-1])
            losses_with.append(self._fit_mlp(self.features, self.targets, mlp, feature_mask))
        
        # MI ∝ Loss_without - Loss_with (higher MI = lower loss with feature)
        return np.array(losses_without) - np.array(losses_with)
    
    def estimate_feature_feature_mi(self, foi_idx: int, fnoi_idx: int) -> float:
        """
        Estimate mutual information between two features I(X_foi; X_fnoi).
        
        Used to identify redundant feature pairs.
        """
        # Use FOI values as target
        foi_mask = torch.zeros(self.n_features, device=device)
        foi_mask[foi_idx] = 1
        foi_values = self.features * foi_mask
        
        # Mask for FNOI only
        fnoi_mask = torch.zeros(self.n_features, device=device)
        fnoi_mask[fnoi_idx] = 1
        
        # Mask for nothing
        zero_mask = torch.zeros(self.n_features, device=device)
        
        losses_without = []
        losses_with = []
        
        for _ in range(3):
            mlp = self._create_mlp(self.n_features, self.n_features)
            losses_without.append(self._fit_mlp(self.features, foi_values, mlp, zero_mask))
            
            mlp = self._create_mlp(self.n_features, self.n_features)
            losses_with.append(self._fit_mlp(self.features, foi_values, mlp, fnoi_mask))
        
        mi_values = np.array(losses_without) - np.array(losses_with)
        return np.mean(mi_values)
    
    def compute_theta(self, foi_idx: int, fnoi_idx: int) -> np.ndarray:
        """
        Compute theta - the change in FOI's contribution when FNOI is present vs absent.
        
        theta < 0: Redundancy (FNOI already provides FOI's info)
        theta > 0: Synergy (FOI + FNOI together provide more info)
        theta ≈ 0: Independence
        """
        # Mask for all features except FOI
        without_foi_mask = torch.ones(self.n_features, device=device)
        without_foi_mask[foi_idx] = 0
        
        # Mask for all features
        all_mask = torch.ones(self.n_features, device=device)
        
        # Mask without both FOI and FNOI
        without_both_mask = torch.ones(self.n_features, device=device)
        without_both_mask[foi_idx] = 0
        without_both_mask[fnoi_idx] = 0
        
        # Mask without FNOI only
        without_fnoi_mask = torch.ones(self.n_features, device=device)
        without_fnoi_mask[fnoi_idx] = 0
        
        n_runs = 3
        
        # Contribution of FOI when FNOI is present: L(all \ FOI) - L(all)
        losses_without_foi = []
        losses_all = []
        for _ in range(n_runs):
            mlp = self._create_mlp(self.n_features, self.targets.shape[-1])
            losses_without_foi.append(self._fit_mlp(self.features, self.targets, mlp, without_foi_mask))
            
            mlp = self._create_mlp(self.n_features, self.targets.shape[-1])
            losses_all.append(self._fit_mlp(self.features, self.targets, mlp, all_mask))
        
        contrib_with_fnoi = np.array(losses_without_foi) - np.array(losses_all)
        
        # Contribution of FOI when FNOI is absent: L(all \ {FOI, FNOI}) - L(all \ FNOI)
        losses_without_both = []
        losses_without_fnoi = []
        for _ in range(n_runs):
            mlp = self._create_mlp(self.n_features, self.targets.shape[-1])
            losses_without_both.append(self._fit_mlp(self.features, self.targets, mlp, without_both_mask))
            
            mlp = self._create_mlp(self.n_features, self.targets.shape[-1])
            losses_without_fnoi.append(self._fit_mlp(self.features, self.targets, mlp, without_fnoi_mask))
        
        contrib_without_fnoi = np.array(losses_without_both) - np.array(losses_without_fnoi)
        
        # Theta = contribution with FNOI - contribution without FNOI
        theta = contrib_with_fnoi - contrib_without_fnoi
        return theta
    
    def analyze_feature(self, foi_idx: int) -> PIDFResult:
        """
        Perform complete PID analysis for a single feature.
        
        Returns:
        - PIDFResult with all PID components
        """
        print(f"\n  Analyzing {self.feature_names[foi_idx]} (index {foi_idx})...")
        
        # 1. Compute individual MI
        mi_values = self.estimate_mi([foi_idx])
        mi_mean, mi_ci = self._calculate_confidence_interval(mi_values)
        print(f"    MI: {mi_mean:.4f} [{mi_ci[0]:.4f}, {mi_ci[1]:.4f}]")
        
        # 2. Analyze interactions with other features
        redundant_with = []
        synergistic_with = []
        total_redundancy = 0
        total_synergy_contribution = 0
        
        for fnoi_idx in range(self.n_features):
            if fnoi_idx == foi_idx:
                continue
            
            # Check feature-feature MI (for redundancy detection)
            ff_mi = self.estimate_feature_feature_mi(foi_idx, fnoi_idx)
            
            # Compute theta
            theta_values = self.compute_theta(foi_idx, fnoi_idx)
            theta_mean, theta_ci = self._calculate_confidence_interval(theta_values)
            
            if theta_ci[1] < -0.001:  # Significantly negative = redundant
                redundant_with.append(fnoi_idx)
                total_redundancy += abs(theta_mean)
                print(f"    Redundant with {self.feature_names[fnoi_idx]}: θ={theta_mean:.4f}")
            elif theta_ci[0] > 0.001:  # Significantly positive = synergistic
                synergistic_with.append(fnoi_idx)
                total_synergy_contribution += theta_mean
                print(f"    Synergistic with {self.feature_names[fnoi_idx]}: θ={theta_mean:.4f}")
        
        # 3. Compute unique information
        unique_info = max(0, mi_mean - total_redundancy)
        
        # 4. Compute total contribution
        total_contribution = mi_mean + total_synergy_contribution
        
        # 5. Determine if feature should be selected
        is_selected = (mi_ci[0] > 0) and (unique_info > 0.001 or len(synergistic_with) > 0)
        
        result = PIDFResult(
            feature_index=foi_idx,
            feature_name=self.feature_names[foi_idx],
            mutual_information=mi_mean,
            synergy=total_synergy_contribution,
            redundancy=total_redundancy,
            unique_info=unique_info,
            total_contribution=total_contribution,
            is_selected=is_selected,
            redundant_with=redundant_with,
            synergistic_with=synergistic_with
        )
        
        return result
    
    def run_full_analysis(self) -> List[PIDFResult]:
        """
        Run complete PIDF analysis for all features.
        
        Returns:
        - List of PIDFResult for each feature
        """
        print("\n" + "=" * 60)
        print("PIDF Analysis - Full Feature Decomposition")
        print("=" * 60)
        
        self.results = []
        
        for foi_idx in range(self.n_features):
            result = self.analyze_feature(foi_idx)
            self.results.append(result)
        
        # Compute feature-feature MI matrix for visualization
        self.feature_mi_matrix = np.zeros((self.n_features, self.n_features))
        for i in range(self.n_features):
            for j in range(self.n_features):
                if i != j:
                    self.feature_mi_matrix[i, j] = self.estimate_feature_feature_mi(i, j)
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.results
    
    def get_selected_features(self) -> List[int]:
        """Return indices of selected features."""
        return [r.feature_index for r in self.results if r.is_selected]
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        print("\n" + "=" * 60)
        print("PIDF Analysis Summary")
        print("=" * 60)
        
        print("\nPer-Feature PID Decomposition:")
        print("-" * 60)
        print(f"{'Feature':<20} {'MI':>8} {'Unique':>8} {'Redund':>8} {'Synergy':>8} {'Selected':>8}")
        print("-" * 60)
        
        for r in self.results:
            selected_str = "✓" if r.is_selected else "✗"
            print(f"{r.feature_name:<20} {r.mutual_information:>8.4f} {r.unique_info:>8.4f} "
                  f"{r.redundancy:>8.4f} {r.synergy:>8.4f} {selected_str:>8}")
        
        print("-" * 60)
        
        selected = self.get_selected_features()
        print(f"\nSelected Features: {[self.feature_names[i] for i in selected]}")
        
        # Print relationships
        print("\nFeature Relationships:")
        for r in self.results:
            if r.redundant_with:
                print(f"  {r.feature_name} is redundant with: "
                      f"{[self.feature_names[i] for i in r.redundant_with]}")
            if r.synergistic_with:
                print(f"  {r.feature_name} has synergy with: "
                      f"{[self.feature_names[i] for i in r.synergistic_with]}")


def visualize_pidf_results(pidf: EnhancedPIDF, save_path: Optional[str] = None):
    """
    Create visualizations of PIDF analysis results.
    
    Creates:
    1. PID component bar chart for each feature
    2. Feature-feature MI heatmap
    3. Information flow diagram
    """
    results = pidf.results
    n_features = len(results)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. PID Component Stacked Bar Chart
    ax1 = axes[0, 0]
    feature_names = [r.feature_name for r in results]
    unique = [r.unique_info for r in results]
    redundancy = [r.redundancy for r in results]
    synergy = [r.synergy for r in results]
    
    x = np.arange(n_features)
    width = 0.6
    
    bars1 = ax1.bar(x, unique, width, label='Unique Info', color='#2ecc71')
    bars2 = ax1.bar(x, redundancy, width, bottom=unique, label='Redundancy', color='#e74c3c')
    
    # Handle negative synergy by plotting below zero
    synergy_pos = [max(0, s) for s in synergy]
    synergy_neg = [min(0, s) for s in synergy]
    
    bars3 = ax1.bar(x, synergy_pos, width, bottom=np.array(unique)+np.array(redundancy), 
                    label='Synergy (+)', color='#3498db')
    bars4 = ax1.bar(x, synergy_neg, width, label='Synergy (-)', color='#9b59b6', alpha=0.7)
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Information (nats)')
    ax1.set_title('PID Decomposition per Feature')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. MI Comparison Bar Chart
    ax2 = axes[0, 1]
    mi_values = [r.mutual_information for r in results]
    total_contrib = [r.total_contribution for r in results]
    
    x = np.arange(n_features)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, mi_values, width, label='Individual MI', color='#9b59b6')
    bars2 = ax2.bar(x + width/2, total_contrib, width, label='Total Contribution', color='#1abc9c')
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Information (nats)')
    ax2.set_title('Individual MI vs Total Contribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.legend()
    
    # 3. Feature-Feature MI Heatmap
    ax3 = axes[1, 0]
    if pidf.feature_mi_matrix is not None:
        sns.heatmap(pidf.feature_mi_matrix, annot=True, fmt='.3f', 
                    xticklabels=feature_names, yticklabels=feature_names,
                    cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'MI'})
        ax3.set_title('Feature-Feature Mutual Information')
    
    # 4. PID Summary Radar/Text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = "PIDF Analysis Summary\n" + "=" * 40 + "\n\n"
    
    total_mi = sum(mi_values)
    total_unique = sum(unique)
    total_redundancy_sum = sum(redundancy)
    total_synergy = sum([s for s in synergy if s > 0])
    
    summary_text += f"Total Information Metrics:\n"
    summary_text += f"  • Sum of Individual MI: {total_mi:.4f}\n"
    summary_text += f"  • Total Unique Info: {total_unique:.4f}\n"
    summary_text += f"  • Total Redundancy: {total_redundancy_sum:.4f}\n"
    summary_text += f"  • Total Synergy: {total_synergy:.4f}\n\n"
    
    summary_text += "Selected Features:\n"
    for r in results:
        if r.is_selected:
            summary_text += f"  ✓ {r.feature_name}\n"
    
    summary_text += "\nFeature Relationships:\n"
    for r in results:
        if r.synergistic_with:
            partners = [pidf.feature_names[i] for i in r.synergistic_with]
            summary_text += f"  • {r.feature_name} ↔ {partners}: Synergistic\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()
    
    return fig


def load_real_data(subject='VP001', task='nback', include_events=True):
    """
    Load and preprocess real EEG/fNIRS data with event labels.
    """
    print(f"Loading {subject} {task} data...")
    loader = SimultaneousData(subject, task)
    
    # Get signals
    eeg_signal, eeg_fs, eeg_labels = loader.get_eeg_signal()
    hbo, hbr, nirs_fs, nirs_labels = loader.get_nirs_signal()
    
    print(f"EEG: {eeg_signal.shape} @ {eeg_fs} Hz, channels: {len(eeg_labels)}")
    print(f"NIRS HbO: {hbo.shape} @ {nirs_fs} Hz, channels: {len(nirs_labels)}")
    
    # Downsample EEG to match NIRS sampling rate
    downsample_factor = int(eeg_fs / nirs_fs)
    eeg_downsampled = eeg_signal[::downsample_factor, :]
    
    # Trim to same length
    min_length = min(eeg_downsampled.shape[0], hbo.shape[0])
    eeg_downsampled = eeg_downsampled[:min_length, :]
    hbo = hbo[:min_length, :]
    hbr = hbr[:min_length, :]
    
    print(f"After preprocessing: EEG {eeg_downsampled.shape}, HbO {hbo.shape}")
    
    if include_events:
        # Get event labels at the downsampled rate
        event_labels, epoch_info = loader.create_epoch_labels(
            signal_length=min_length,
            fs=nirs_fs,
            modality='nirs'
        )
        print(f"Event labels: {np.sum(event_labels == 1)} target samples, "
              f"{np.sum(event_labels == 0)} baseline samples")
        print(f"Number of epochs: {len(epoch_info['starts'])}")
        
        # Also get cognitive load labels
        load_labels = loader.get_cognitive_load_labels(min_length, nirs_fs, 'nirs')
        epoch_info['load_labels'] = load_labels
        
        return eeg_downsampled, hbo, hbr, event_labels, epoch_info, eeg_labels, nirs_labels
    
    return eeg_downsampled, hbo, hbr


def prepare_multimodal_features(eeg: np.ndarray, hbo: np.ndarray, 
                                 eeg_labels: List[str], nirs_labels: List[str],
                                 mode: str = 'average') -> Tuple[np.ndarray, List[str]]:
    """
    Prepare features for PIDF analysis from multimodal data.
    
    Parameters:
    - eeg: EEG data (n_samples, n_eeg_channels)
    - hbo: fNIRS HbO data (n_samples, n_nirs_channels)
    - eeg_labels: EEG channel labels
    - nirs_labels: fNIRS channel labels
    - mode: 'average' for averaged signals, 'channels' for individual channels,
            'regions' for region-based averaging
    
    Returns:
    - features: np.ndarray (n_samples, n_features)
    - feature_names: List of feature names
    """
    if mode == 'average':
        # Simple: average across all channels per modality
        features = np.column_stack([
            np.mean(eeg, axis=1),
            np.mean(hbo, axis=1),
        ])
        feature_names = ['EEG_avg', 'fNIRS_avg']
        
    elif mode == 'regions':
        # Group channels by brain regions (simplified)
        # EEG: Frontal (F), Central (C), Parietal (P), Occipital (O)
        frontal_eeg = [i for i, l in enumerate(eeg_labels) if l.startswith('F')]
        central_eeg = [i for i, l in enumerate(eeg_labels) if l.startswith('C')]
        parietal_eeg = [i for i, l in enumerate(eeg_labels) if l.startswith('P')]
        occipital_eeg = [i for i, l in enumerate(eeg_labels) if l.startswith('O')]
        
        features_list = []
        names_list = []
        
        if frontal_eeg:
            features_list.append(np.mean(eeg[:, frontal_eeg], axis=1))
            names_list.append('EEG_Frontal')
        if central_eeg:
            features_list.append(np.mean(eeg[:, central_eeg], axis=1))
            names_list.append('EEG_Central')
        if parietal_eeg:
            features_list.append(np.mean(eeg[:, parietal_eeg], axis=1))
            names_list.append('EEG_Parietal')
        if occipital_eeg:
            features_list.append(np.mean(eeg[:, occipital_eeg], axis=1))
            names_list.append('EEG_Occipital')
        
        # fNIRS: Left and Right hemisphere (simplified based on channel names)
        left_nirs = [i for i, l in enumerate(nirs_labels) if '_L' in str(l) or 'L' in str(l)[:2]]
        right_nirs = [i for i, l in enumerate(nirs_labels) if '_R' in str(l) or 'R' in str(l)[:2]]
        
        if left_nirs:
            features_list.append(np.mean(hbo[:, left_nirs], axis=1))
            names_list.append('fNIRS_Left')
        if right_nirs:
            features_list.append(np.mean(hbo[:, right_nirs], axis=1))
            names_list.append('fNIRS_Right')
        
        # If region extraction failed, fall back to average
        if len(features_list) < 2:
            features_list = [np.mean(eeg, axis=1), np.mean(hbo, axis=1)]
            names_list = ['EEG_avg', 'fNIRS_avg']
        
        features = np.column_stack(features_list)
        feature_names = names_list
        
    else:  # mode == 'channels'
        # Use top channels (limited for computational tractability)
        max_channels = 5
        eeg_subset = eeg[:, :min(max_channels, eeg.shape[1])]
        hbo_subset = hbo[:, :min(max_channels, hbo.shape[1])]
        
        features = np.column_stack([eeg_subset, hbo_subset])
        feature_names = ([f'EEG_{eeg_labels[i]}' for i in range(eeg_subset.shape[1])] + 
                        [f'fNIRS_{nirs_labels[i]}' for i in range(hbo_subset.shape[1])])
    
    return features, feature_names


def main():
    """Main analysis function using Enhanced PIDF."""
    print("=" * 70)
    print("EEG/fNIRS Partial Information Decomposition Analysis (PIDF Method)")
    print("=" * 70)
    
    # Load data
    eeg, hbo, hbr, event_labels, epoch_info, eeg_labels, nirs_labels = load_real_data(
        include_events=True
    )
    
    # =========================================================================
    # Analysis 1: Basic Bimodal Analysis (EEG vs fNIRS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Basic Bimodal PIDF (EEG avg vs fNIRS avg)")
    print("=" * 70)
    
    features, feature_names = prepare_multimodal_features(
        eeg, hbo, eeg_labels, nirs_labels, mode='average'
    )
    
    # Subsample for computational tractability
    max_samples = 3000
    if len(event_labels) > max_samples:
        print(f"\nSubsampling from {len(event_labels)} to {max_samples} samples...")
        np.random.seed(42)
        indices = np.random.choice(len(event_labels), max_samples, replace=False)
        features_sub = features[indices]
        targets_sub = event_labels[indices]
    else:
        features_sub = features
        targets_sub = event_labels
    
    print(f"\nFeatures: {feature_names}")
    print(f"Target: Event labels (0=baseline, 1=target)")
    
    # Run Enhanced PIDF
    pidf = EnhancedPIDF(
        features_sub, targets_sub, 
        feature_names=feature_names,
        num_iterations=150
    )
    
    results = pidf.run_full_analysis()
    pidf.print_summary()
    
    # Visualize
    viz_path = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'pidf_bimodal.png')
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    visualize_pidf_results(pidf, save_path=viz_path)
    
    # =========================================================================
    # Analysis 2: Region-based Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Region-based PIDF")
    print("=" * 70)
    
    features_regions, region_names = prepare_multimodal_features(
        eeg, hbo, eeg_labels, nirs_labels, mode='regions'
    )
    
    print(f"\nFeatures: {region_names}")
    
    if len(event_labels) > max_samples:
        features_regions_sub = features_regions[indices]
    else:
        features_regions_sub = features_regions
    
    pidf_regions = EnhancedPIDF(
        features_regions_sub, targets_sub,
        feature_names=region_names,
        num_iterations=150
    )
    
    results_regions = pidf_regions.run_full_analysis()
    pidf_regions.print_summary()
    
    # Visualize
    viz_path_regions = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'pidf_regions.png')
    visualize_pidf_results(pidf_regions, save_path=viz_path_regions)
    
    # =========================================================================
    # Analysis 3: Cognitive Load Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Cognitive Load Comparison")
    print("=" * 70)
    
    load_labels = epoch_info['load_labels']
    unique_loads = np.unique(load_labels[load_labels > 0])
    
    load_results = {}
    
    for load_level in unique_loads:
        mask = load_labels == load_level
        if np.sum(mask) > 500:  # Enough samples
            print(f"\n--- {load_level}-back Task ---")
            
            features_load = features[mask]
            targets_load = event_labels[mask]
            
            # Subsample if needed
            if len(targets_load) > 2000:
                np.random.seed(42)
                idx = np.random.choice(len(targets_load), 2000, replace=False)
                features_load = features_load[idx]
                targets_load = targets_load[idx]
            
            if len(np.unique(targets_load)) > 1:  # Need both classes
                pidf_load = EnhancedPIDF(
                    features_load, targets_load,
                    feature_names=feature_names,
                    num_iterations=100
                )
                
                load_results[load_level] = pidf_load.run_full_analysis()
                pidf_load.print_summary()
    
    # =========================================================================
    # Save Results Summary
    # =========================================================================
    results_file = os.path.join(os.path.dirname(__file__), '..', 'result', 'pidf_analysis_results.txt')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("PIDF Analysis Results for EEG/fNIRS Data\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. Basic Bimodal Analysis:\n")
        f.write("-" * 40 + "\n")
        for r in results:
            f.write(f"  {r.feature_name}:\n")
            f.write(f"    MI: {r.mutual_information:.4f}\n")
            f.write(f"    Unique: {r.unique_info:.4f}\n")
            f.write(f"    Redundancy: {r.redundancy:.4f}\n")
            f.write(f"    Synergy: {r.synergy:.4f}\n")
            f.write(f"    Selected: {r.is_selected}\n")
        
        f.write(f"\n  Selected Features: {[r.feature_name for r in results if r.is_selected]}\n")
        
        f.write("\n\n2. Region-based Analysis:\n")
        f.write("-" * 40 + "\n")
        for r in results_regions:
            f.write(f"  {r.feature_name}:\n")
            f.write(f"    MI: {r.mutual_information:.4f}\n")
            f.write(f"    Unique: {r.unique_info:.4f}\n")
            f.write(f"    Redundancy: {r.redundancy:.4f}\n")
            f.write(f"    Synergy: {r.synergy:.4f}\n")
        
        f.write("\n\n3. Cognitive Load Comparison:\n")
        f.write("-" * 40 + "\n")
        for load_level, load_res in load_results.items():
            f.write(f"\n  {load_level}-back task:\n")
            for r in load_res:
                f.write(f"    {r.feature_name}: MI={r.mutual_information:.4f}, "
                       f"Unique={r.unique_info:.4f}, Synergy={r.synergy:.4f}\n")
    
    print(f"\n\nResults saved to: {results_file}")
    print("\n" + "=" * 70)
    print("PIDF ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
