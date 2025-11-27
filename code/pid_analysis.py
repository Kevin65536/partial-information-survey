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
from typing import List, Dict, Tuple, Optional, Any

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
    synergy: float  # Synergistic contribution (positive theta)
    redundancy: float  # Redundant contribution (negative theta)
    unique_info: float  # Unique information = MI - Redundancy
    total_contribution: float  # MI + Synergy - Redundancy
    is_selected: bool
    redundant_with: List[int]  # Features this is redundant with
    synergistic_with: List[int]  # Features this has synergy with
    # New fields for detailed analysis
    theta_values: Dict[int, float] = None  # Raw theta values for each feature pair
    theta_ci: Dict[int, Tuple[float, float]] = None  # Confidence intervals
    interaction_type: str = "independent"  # 'synergistic', 'redundant', 'independent', 'mixed'
    
    def __post_init__(self):
        if self.theta_values is None:
            self.theta_values = {}
        if self.theta_ci is None:
            self.theta_ci = {}


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
        - PIDFResult with all PID components including raw theta values
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
        theta_values_dict = {}  # Store raw theta values
        theta_ci_dict = {}  # Store confidence intervals
        
        for fnoi_idx in range(self.n_features):
            if fnoi_idx == foi_idx:
                continue
            
            # Check feature-feature MI (for redundancy detection)
            ff_mi = self.estimate_feature_feature_mi(foi_idx, fnoi_idx)
            
            # Compute theta with more runs for stability
            theta_values = self.compute_theta(foi_idx, fnoi_idx)
            theta_mean, theta_ci = self._calculate_confidence_interval(theta_values)
            
            # Store raw values
            theta_values_dict[fnoi_idx] = theta_mean
            theta_ci_dict[fnoi_idx] = theta_ci
            
            # Determine relationship type based on theta
            # Use relative threshold: |θ| > 0.1 * MI or absolute > 0.001
            threshold = max(0.001, 0.1 * abs(mi_mean))
            
            if theta_ci[1] < -threshold:  # Significantly negative = redundant/interference
                redundant_with.append(fnoi_idx)
                total_redundancy += abs(theta_mean)
                print(f"    ⊖ Redundant/Interference with {self.feature_names[fnoi_idx]}: θ={theta_mean:.4f} (CI: [{theta_ci[0]:.4f}, {theta_ci[1]:.4f}])")
            elif theta_ci[0] > threshold:  # Significantly positive = synergistic
                synergistic_with.append(fnoi_idx)
                total_synergy_contribution += theta_mean
                print(f"    ⊕ Synergistic with {self.feature_names[fnoi_idx]}: θ={theta_mean:.4f} (CI: [{theta_ci[0]:.4f}, {theta_ci[1]:.4f}])")
            else:  # Not significant but still report
                relation = "slightly redundant" if theta_mean < 0 else "slightly synergistic" if theta_mean > 0 else "independent"
                print(f"    ○ {relation.capitalize()} with {self.feature_names[fnoi_idx]}: θ={theta_mean:.4f} (CI: [{theta_ci[0]:.4f}, {theta_ci[1]:.4f}]) [not significant]")
        
        # 3. Compute unique information
        unique_info = max(0, mi_mean - total_redundancy)
        
        # 4. Compute total contribution (can be negative if redundancy dominates)
        total_contribution = mi_mean + total_synergy_contribution - total_redundancy
        
        # 5. Determine interaction type
        if len(synergistic_with) > 0 and len(redundant_with) > 0:
            interaction_type = "mixed"
        elif len(synergistic_with) > 0:
            interaction_type = "synergistic"
        elif len(redundant_with) > 0:
            interaction_type = "redundant"
        else:
            # Check if there's a trend even if not significant
            avg_theta = np.mean(list(theta_values_dict.values())) if theta_values_dict else 0
            if avg_theta > 0.0005:
                interaction_type = "weakly_synergistic"
            elif avg_theta < -0.0005:
                interaction_type = "weakly_redundant"
            else:
                interaction_type = "independent"
        
        # 6. Determine if feature should be selected
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
            synergistic_with=synergistic_with,
            theta_values=theta_values_dict,
            theta_ci=theta_ci_dict,
            interaction_type=interaction_type
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


def visualize_pidf_paper_style(results: List[PIDFResult], feature_names: List[str],
                                title: str = "PIDF Decomposition",
                                save_path: Optional[str] = None,
                                ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
    """
    Create visualization following PIDF paper style with support for redundancy/interference.
    
    According to the paper:
    - Red: I(Y; F_i) - Individual mutual information
    - Green: FWS(Y; F_{\\i}, F_i) - Feature-wise synergy (协同信息, θ > 0)
    - Purple: FWR(Y | F_{\\i}, F_i) - Feature-wise redundancy/interference (冗余/干扰, θ < 0)
    
    Enhanced to show:
    - Positive θ (synergy) stacked above MI
    - Negative θ (redundancy/interference) shown below zero line
    - Interaction type annotation for each feature
    """
    n_features = len(results)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(3 + n_features * 2, 7))
        standalone = True
    else:
        fig = None
        standalone = False
    
    # Extract components
    mi_values = [r.mutual_information for r in results]
    synergy_values = [r.synergy for r in results]  # Positive θ contributions
    redundancy_values = [r.redundancy for r in results]  # |Negative θ| contributions
    interaction_types = [r.interaction_type for r in results]
    
    # Get raw theta values for display
    theta_displays = []
    for r in results:
        if r.theta_values:
            avg_theta = np.mean(list(r.theta_values.values()))
            theta_displays.append(avg_theta)
        else:
            theta_displays.append(0)
    
    x = np.arange(n_features)
    width = 0.6
    
    # Colors
    color_mi = '#E74C3C'  # Red - I(Y; F_i)
    color_synergy = '#2ECC71'  # Green - Synergy (θ > 0)
    color_redundancy = '#9B59B6'  # Purple - Redundancy (θ < 0)
    color_interference = '#E67E22'  # Orange - Interference indicator
    
    # Draw MI (red) as base - always positive
    bars_mi = ax.bar(x, mi_values, width, label=r'$I(Y; F_i)$ - Individual MI', 
                     color=color_mi, edgecolor='black', linewidth=1)
    
    # Draw Synergy (green) on top of MI - only positive values
    synergy_pos = [max(0, s) for s in synergy_values]
    if any(s > 0 for s in synergy_pos):
        bars_synergy = ax.bar(x, synergy_pos, width, bottom=mi_values,
                              label='Synergy (theta > 0)', 
                              color=color_synergy, edgecolor='black', linewidth=1)
    
    # Draw Redundancy (purple) BELOW zero line - shows interference effect
    redundancy_neg = [-r for r in redundancy_values]  # Make negative for display
    if any(r > 0 for r in redundancy_values):
        bars_redundancy = ax.bar(x, redundancy_neg, width,
                                  label='Redundancy/Interference (theta < 0)', 
                                  color=color_redundancy, edgecolor='black', linewidth=1,
                                  hatch='///')
    
    # Add annotations
    for i in range(n_features):
        mi = mi_values[i]
        syn = synergy_pos[i]
        red = redundancy_values[i]
        theta = theta_displays[i]
        itype = interaction_types[i]
        
        # Top annotation: MCI value
        mci = mi + syn
        ax.annotate(f'MCI={mci:.4f}', xy=(x[i], mci + 0.001), ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        # MI value inside bar
        if mi > 0.002:
            ax.annotate(f'{mi:.4f}', xy=(x[i], mi/2), ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
        
        # Bottom annotation: redundancy/interference
        if red > 0:
            ax.annotate(f'-{red:.4f}', xy=(x[i], -red - 0.001), ha='center', va='top',
                        fontsize=8, color=color_redundancy)
        
        # Interaction type symbol
        type_symbols = {
            'synergistic': '⊕',
            'redundant': '⊖',
            'independent': '○',
            'mixed': '◐',
            'weakly_synergistic': '(+)',
            'weakly_redundant': '(-)',
        }
        symbol = type_symbols.get(itype, '?')
        
        # Color based on theta
        symbol_color = color_synergy if theta > 0 else color_redundancy if theta < 0 else 'gray'
        ax.annotate(f'{symbol}\nθ={theta:.4f}', xy=(x[i], -max(redundancy_values) - 0.003 if redundancy_values else -0.003), 
                    ha='center', va='top', fontsize=8, color=symbol_color)
    
    # Styling
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Information (Nats)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r.feature_name for r in results], rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Set y limits to show both positive and negative regions
    max_pos = max(mi + syn for mi, syn in zip(mi_values, synergy_pos)) * 1.3
    max_neg = max(redundancy_values) * 1.5 if redundancy_values and max(redundancy_values) > 0 else 0.005
    ax.set_ylim(bottom=-max_neg, top=max_pos)
    ax.grid(axis='y', alpha=0.3)
    
    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
        plt.close()
    
    return fig


def visualize_nback_comparison(load_results: Dict[int, List[PIDFResult]], 
                               feature_names: List[str],
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create enhanced comparison visualization across different n-back conditions.
    
    Shows how PID components (including redundancy/interference) change with cognitive load.
    """
    n_conditions = len(load_results)
    n_features = len(feature_names)
    
    if n_conditions == 0:
        print("No results to visualize")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Layout: 
    # Top row: Individual PIDF plots for each n-back condition
    # Middle: Theta value comparison (key for synergy/redundancy)
    # Bottom: Trend analysis
    
    sorted_loads = sorted(load_results.keys())
    
    gs = fig.add_gridspec(3, n_conditions + 1, height_ratios=[1.2, 1, 1], 
                          width_ratios=[1] * n_conditions + [1.2])
    
    # Top row: Individual PIDF decomposition per condition
    for col, load_level in enumerate(sorted_loads):
        ax = fig.add_subplot(gs[0, col])
        results = load_results[load_level]
        visualize_pidf_paper_style(
            results, feature_names[:len(results)], 
            title=f'{load_level}-back Task',
            ax=ax
        )
    
    # Top right: Summary metrics comparison
    ax_summary = fig.add_subplot(gs[0, n_conditions])
    
    metrics = {'MI': [], 'Synergy': [], 'Redundancy': [], 'Net Effect': []}
    labels = []
    interaction_summary = []
    
    for load_level in sorted_loads:
        results = load_results[load_level]
        total_mi = sum(r.mutual_information for r in results)
        total_syn = sum(r.synergy for r in results)
        total_red = sum(r.redundancy for r in results)
        net_effect = total_syn - total_red  # Positive = synergistic, Negative = redundant
        
        metrics['MI'].append(total_mi)
        metrics['Synergy'].append(total_syn)
        metrics['Redundancy'].append(-total_red)  # Show as negative
        metrics['Net Effect'].append(net_effect)
        labels.append(f'{load_level}-back')
        
        # Determine overall interaction type
        if net_effect > 0.001:
            interaction_summary.append('Synergistic')
        elif net_effect < -0.001:
            interaction_summary.append('Redundant')
        else:
            interaction_summary.append('Independent')
    
    x = np.arange(len(labels))
    width = 0.2
    
    colors = {'MI': '#E74C3C', 'Synergy': '#2ECC71', 'Redundancy': '#9B59B6', 'Net Effect': '#3498DB'}
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        ax_summary.bar(x + offset, values, width, label=metric_name, 
                       color=colors[metric_name], edgecolor='black', linewidth=0.5)
    
    ax_summary.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax_summary.set_xlabel('Cognitive Load')
    ax_summary.set_ylabel('Information (Nats)')
    ax_summary.set_title('PID Metrics Summary', fontweight='bold')
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(labels)
    ax_summary.legend(loc='upper right', fontsize=8)
    ax_summary.grid(axis='y', alpha=0.3)
    
    # Add interaction type annotations
    for i, itype in enumerate(interaction_summary):
        color = '#2ECC71' if itype == 'Synergistic' else '#9B59B6' if itype == 'Redundant' else 'gray'
        ax_summary.annotate(itype, xy=(x[i], ax_summary.get_ylim()[1] * 0.9), 
                           ha='center', fontsize=9, color=color, fontweight='bold')
    
    # Middle row: Theta values comparison across conditions
    ax_theta = fig.add_subplot(gs[1, :])
    
    theta_data = {}
    for feat_idx, feat_name in enumerate(feature_names):
        theta_by_load = []
        for load_level in sorted_loads:
            results = load_results[load_level]
            if feat_idx < len(results):
                r = results[feat_idx]
                if r.theta_values:
                    avg_theta = np.mean(list(r.theta_values.values()))
                else:
                    avg_theta = 0
                theta_by_load.append(avg_theta)
        if theta_by_load:
            theta_data[feat_name] = theta_by_load
    
    x_positions = np.arange(len(sorted_loads))
    width_per_feature = 0.8 / len(theta_data)
    colors_features = plt.cm.Set2(np.linspace(0, 1, len(theta_data)))
    
    for feat_idx, (feat_name, theta_vals) in enumerate(theta_data.items()):
        offset = (feat_idx - len(theta_data)/2 + 0.5) * width_per_feature
        bars = ax_theta.bar(x_positions + offset, theta_vals, width_per_feature * 0.9,
                            label=feat_name, color=colors_features[feat_idx], 
                            edgecolor='black', linewidth=0.5)
        
        # Add value annotations
        for i, val in enumerate(theta_vals):
            color = '#2ECC71' if val > 0 else '#9B59B6'
            ax_theta.annotate(f'{val:.4f}', xy=(x_positions[i] + offset, val),
                             ha='center', va='bottom' if val > 0 else 'top',
                             fontsize=8, color=color)
    
    ax_theta.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax_theta.axhspan(-0.001, 0.001, alpha=0.2, color='gray', label='Threshold')
    ax_theta.set_xlabel('Cognitive Load Condition', fontsize=12)
    ax_theta.set_ylabel('θ Value (Synergy - Redundancy)', fontsize=12)
    ax_theta.set_title('Feature Interaction θ Values Across N-back Conditions\n(θ>0: Synergy, θ<0: Redundancy/Interference)', 
                       fontsize=12, fontweight='bold')
    ax_theta.set_xticks(x_positions)
    ax_theta.set_xticklabels([f'{l}-back' for l in sorted_loads], fontsize=11)
    ax_theta.legend(loc='upper right', fontsize=9)
    ax_theta.grid(axis='y', alpha=0.3)
    
    # Bottom row: Net contribution trend
    ax_trend = fig.add_subplot(gs[2, :])
    
    for feat_idx, feat_name in enumerate(feature_names):
        mi_by_load = []
        net_contrib = []  # MI + Synergy - Redundancy
        
        for load_level in sorted_loads:
            results = load_results[load_level]
            if feat_idx < len(results):
                r = results[feat_idx]
                mi_by_load.append(r.mutual_information)
                net_contrib.append(r.total_contribution)
        
        if mi_by_load:
            ax_trend.plot(sorted_loads, mi_by_load, 'o--', 
                         label=f'{feat_name} (MI)', color=colors_features[feat_idx], alpha=0.5)
            ax_trend.plot(sorted_loads, net_contrib, 's-', 
                         label=f'{feat_name} (Net)', color=colors_features[feat_idx], linewidth=2)
    
    ax_trend.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax_trend.set_xlabel('N-back Level', fontsize=12)
    ax_trend.set_ylabel('Information Contribution (Nats)', fontsize=12)
    ax_trend.set_title('Information Contribution Trends: MI vs Net (MI + Synergy - Redundancy)', 
                       fontsize=12, fontweight='bold')
    ax_trend.set_xticks(sorted_loads)
    ax_trend.set_xticklabels([f'{l}-back' for l in sorted_loads])
    ax_trend.legend(loc='upper right', fontsize=8, ncol=2)
    ax_trend.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nN-back comparison saved to: {save_path}")
    
    plt.close()
    return fig


def visualize_pidf_results(pidf: EnhancedPIDF, save_path: Optional[str] = None):
    """
    Create visualizations of PIDF analysis results in paper style.
    
    Creates:
    1. PID component stacked bar chart (paper style)
    2. Feature-feature MI heatmap
    3. Information breakdown pie chart
    """
    results = pidf.results
    n_features = len(results)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    
    # 1. Main PIDF Paper-style Bar Chart (larger, top spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    
    feature_names = [r.feature_name for r in results]
    mi_values = [r.mutual_information for r in results]
    synergy_values = [r.synergy for r in results]
    redundancy_values = [r.redundancy for r in results]
    
    x = np.arange(n_features)
    width = 0.5
    
    # Paper style colors
    color_mi = '#E74C3C'  # Red - I(Y; F_i)
    color_synergy = '#2ECC71'  # Green - FWS
    color_redundancy = '#9B59B6'  # Purple - FWR
    
    # Main stacked bar: MI + Synergy
    bars_mi = ax1.bar(x, mi_values, width, label=r'$I(Y; F_i)$ - Individual MI', 
                      color=color_mi, edgecolor='black', linewidth=1)
    bars_synergy = ax1.bar(x, synergy_values, width, bottom=mi_values,
                           label=r'$FWS(Y; \mathcal{F}_{\backslash i}, F_i)$ - Synergy',
                           color=color_synergy, edgecolor='black', linewidth=1)
    
    # Show redundancy as small bars beside main bars
    for i, red in enumerate(redundancy_values):
        if red > 0.0001:
            ax1.bar(x[i] + width * 0.4, red, width * 0.25, 
                    color=color_redundancy, edgecolor='black', linewidth=1,
                    hatch='///')
    
    # Add dummy for legend
    if any(r > 0.0001 for r in redundancy_values):
        ax1.bar([], [], width, label=r'$FWR$ - Redundancy', 
                color=color_redundancy, hatch='///', edgecolor='black')
    
    # Add MCI annotations (MCI = MI + Synergy)
    for i, (mi, syn) in enumerate(zip(mi_values, synergy_values)):
        mci = mi + syn
        ax1.annotate(f'MCI={mci:.4f}', xy=(x[i], mci), ha='center', va='bottom', 
                     fontsize=10, fontweight='bold')
        ax1.annotate(f'MI={mi:.4f}', xy=(x[i], mi/2), ha='center', va='center',
                     fontsize=9, color='white')
    
    # Add bracket for OCI if synergy exists
    for i, (mi, syn) in enumerate(zip(mi_values, synergy_values)):
        if syn > 0.001:
            ax1.annotate('', xy=(x[i] + width/2 + 0.05, mi), xytext=(x[i] + width/2 + 0.05, mi + syn),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            ax1.annotate(f'OCI', xy=(x[i] + width/2 + 0.1, mi + syn/2), fontsize=8, va='center')
    
    ax1.set_xlabel('Features', fontsize=14)
    ax1.set_ylabel('Information (Nats)', fontsize=14)
    ax1.set_title('Interpretable Output of PIDF - Paper Style Visualization', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names, fontsize=12)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Feature-Feature MI Heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    if pidf.feature_mi_matrix is not None:
        mask = np.eye(n_features, dtype=bool)
        sns.heatmap(pidf.feature_mi_matrix, annot=True, fmt='.4f', 
                    xticklabels=feature_names, yticklabels=feature_names,
                    cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'MI'},
                    mask=mask, linewidths=0.5)
        ax2.set_title('Feature-Feature Mutual Information', fontsize=12, fontweight='bold')
    
    # 3. Information Breakdown Summary
    ax3 = fig.add_subplot(gs[1, 1])
    
    total_mi = sum(mi_values)
    total_unique = sum(r.unique_info for r in results)
    total_redundancy = sum(redundancy_values)
    total_synergy = sum(s for s in synergy_values if s > 0)
    
    # Pie chart of information components
    sizes = [total_unique, total_synergy, total_redundancy]
    labels_pie = [f'Unique\n{total_unique:.4f}', 
                  f'Synergy\n{total_synergy:.4f}', 
                  f'Redundancy\n{total_redundancy:.4f}']
    colors_pie = ['#3498DB', '#2ECC71', '#9B59B6']
    explode = (0.05, 0.05, 0.05)
    
    # Only show non-zero slices
    nonzero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels_pie, colors_pie, explode) if s > 0.0001]
    if nonzero:
        sizes_nz, labels_nz, colors_nz, explode_nz = zip(*nonzero)
        wedges, texts, autotexts = ax3.pie(sizes_nz, labels=labels_nz, colors=colors_nz,
                                            explode=explode_nz, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Information Decomposition Breakdown', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No significant information detected', 
                 ha='center', va='center', fontsize=12)
        ax3.set_title('Information Decomposition Breakdown', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.close()
    
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


# =============================================================================
# Advanced Feature Extraction Functions
# =============================================================================

# Frequency band definitions for EEG
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
}


def extract_band_power(signal: np.ndarray, fs: float, band: Tuple[float, float], 
                       window_size: int = None) -> np.ndarray:
    """
    Extract power in a specific frequency band using FFT.
    
    Parameters:
    - signal: (n_samples, n_channels) or (n_samples,)
    - fs: Sampling frequency
    - band: (low_freq, high_freq) tuple
    - window_size: Size of sliding window (default: 1 second)
    
    Returns:
    - power: Band power for each sample (smoothed)
    """
    from scipy.signal import welch
    from scipy.ndimage import uniform_filter1d
    
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    n_samples, n_channels = signal.shape
    
    if window_size is None:
        window_size = int(fs)  # 1 second window
    
    # Compute power using sliding window approach
    powers = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Use uniform filter to smooth the squared signal as approximation
        # This is faster than computing full spectral analysis per sample
        filtered = signal[:, ch] ** 2
        powers[:, ch] = uniform_filter1d(filtered, size=window_size, mode='nearest')
    
    # For more accurate band power, we can use the variance in that band
    # But for efficiency, we'll use a simpler approach based on signal energy
    
    return np.mean(powers, axis=1) if n_channels > 1 else powers.flatten()


def extract_eeg_features(eeg: np.ndarray, eeg_labels: List[str], fs: float = 10.0,
                         mode: str = 'minimal') -> Tuple[np.ndarray, List[str]]:
    """
    Extract EEG features with configurable complexity.
    
    Parameters:
    - eeg: EEG data (n_samples, n_channels), already downsampled
    - eeg_labels: Channel names
    - fs: Sampling frequency (after downsampling)
    - mode: 'minimal' (1 feature), 'basic' (2-3 features), 'moderate' (4-5 features)
    
    Returns:
    - features: (n_samples, n_features)
    - feature_names: List of feature names
    """
    features_list = []
    names_list = []
    
    # Global average - always included
    eeg_avg = np.mean(eeg, axis=1)
    features_list.append(eeg_avg)
    names_list.append('EEG_avg')
    
    if mode == 'minimal':
        # Just global average
        pass
    
    elif mode == 'basic':
        # Add variance as proxy for activation level
        from scipy.ndimage import uniform_filter1d
        window = max(int(fs), 3)
        global_var = uniform_filter1d(np.var(eeg, axis=1), size=window, mode='nearest')
        features_list.append(global_var)
        names_list.append('EEG_power')
    
    elif mode == 'moderate':
        from scipy.ndimage import uniform_filter1d
        window = max(int(fs), 3)
        
        # Frontal activity (related to working memory)
        frontal_idx = [i for i, l in enumerate(eeg_labels) if l.startswith('F')]
        if frontal_idx:
            frontal_avg = np.mean(eeg[:, frontal_idx], axis=1)
            features_list.append(frontal_avg)
            names_list.append('EEG_Frontal')
        
        # Global power
        global_var = uniform_filter1d(np.var(eeg, axis=1), size=window, mode='nearest')
        features_list.append(global_var)
        names_list.append('EEG_power')
    
    features = np.column_stack(features_list)
    return features, names_list


def extract_fnirs_features(hbo: np.ndarray, hbr: np.ndarray, nirs_labels: List[str], 
                           fs: float = 10.0, mode: str = 'minimal') -> Tuple[np.ndarray, List[str]]:
    """
    Extract fNIRS features with configurable complexity.
    
    Parameters:
    - hbo: HbO data (n_samples, n_channels)
    - hbr: HbR data (n_samples, n_channels)
    - nirs_labels: Channel names
    - fs: Sampling frequency
    - mode: 'minimal' (1 feature), 'basic' (2 features), 'moderate' (3-4 features)
    
    Returns:
    - features: (n_samples, n_features)
    - feature_names: List of feature names
    """
    features_list = []
    names_list = []
    
    # Global HbO average - always included
    hbo_avg = np.mean(hbo, axis=1)
    features_list.append(hbo_avg)
    names_list.append('fNIRS_avg')
    
    if mode == 'minimal':
        # Just HbO average
        pass
    
    elif mode == 'basic':
        # Add oxygenation index (HbO - HbR)
        if hbr is not None:
            hbr_avg = np.mean(hbr, axis=1)
            oxy_index = hbo_avg - hbr_avg
            features_list.append(oxy_index)
            names_list.append('fNIRS_Oxy')
    
    elif mode == 'moderate':
        if hbr is not None:
            hbr_avg = np.mean(hbr, axis=1)
            # Oxygenation index
            oxy_index = hbo_avg - hbr_avg
            features_list.append(oxy_index)
            names_list.append('fNIRS_Oxy')
        
        # Temporal derivative (hemodynamic response dynamics)
        hbo_slope = np.gradient(hbo_avg)
        features_list.append(hbo_slope)
        names_list.append('fNIRS_slope')
    
    features = np.column_stack(features_list)
    return features, names_list


def prepare_advanced_features(eeg: np.ndarray, hbo: np.ndarray, hbr: np.ndarray,
                              eeg_labels: List[str], nirs_labels: List[str],
                              fs: float = 10.0, mode: str = 'simple') -> Tuple[np.ndarray, List[str]]:
    """
    Prepare multimodal features for PIDF analysis.
    
    Parameters:
    - eeg: EEG data (n_samples, n_channels)
    - hbo: fNIRS HbO data (n_samples, n_channels)
    - hbr: fNIRS HbR data (n_samples, n_channels)
    - eeg_labels, nirs_labels: Channel labels
    - fs: Sampling frequency
    - mode: 'simple' (2 features), 'basic' (4 features), 'moderate' (6 features)
    
    Returns:
    - features: Combined feature matrix
    - feature_names: Feature names
    
    Note: Fewer features = faster computation. For PID analysis, we focus on
    the core question: does combining EEG + fNIRS provide synergistic information?
    """
    if mode == 'simple':
        # Minimal: just averaged signals (2 features total)
        # This is sufficient for testing basic synergy between modalities
        eeg_feats, eeg_names = extract_eeg_features(eeg, eeg_labels, fs, mode='minimal')
        fnirs_feats, fnirs_names = extract_fnirs_features(hbo, hbr, nirs_labels, fs, mode='minimal')
        features = np.column_stack([eeg_feats, fnirs_feats])
        return features, eeg_names + fnirs_names
    
    elif mode == 'basic':
        # Basic: averaged + power/oxygenation (4 features total)
        eeg_feats, eeg_names = extract_eeg_features(eeg, eeg_labels, fs, mode='basic')
        fnirs_feats, fnirs_names = extract_fnirs_features(hbo, hbr, nirs_labels, fs, mode='basic')
        features = np.column_stack([eeg_feats, fnirs_feats])
        return features, eeg_names + fnirs_names
    
    else:  # moderate
        # Moderate: regional + temporal dynamics (6 features total)
        eeg_feats, eeg_names = extract_eeg_features(eeg, eeg_labels, fs, mode='moderate')
        fnirs_feats, fnirs_names = extract_fnirs_features(hbo, hbr, nirs_labels, fs, mode='moderate')
        features = np.column_stack([eeg_feats, fnirs_feats])
        return features, eeg_names + fnirs_names


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
    """Main analysis function using Enhanced PIDF with advanced features."""
    print("=" * 70)
    print("EEG/fNIRS Partial Information Decomposition Analysis (PIDF Method)")
    print("Enhanced Version with Redundancy/Interference Detection")
    print("=" * 70)
    
    # Load data
    eeg, hbo, hbr, event_labels, epoch_info, eeg_labels, nirs_labels = load_real_data(
        include_events=True
    )
    
    nirs_fs = 10.0  # fNIRS sampling rate after downsampling
    
    # =========================================================================
    # Analysis 1: Basic Bimodal Analysis (EEG vs fNIRS) - Simple features
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Basic Bimodal PIDF (Simple Features)")
    print("=" * 70)
    
    features_simple, feature_names_simple = prepare_advanced_features(
        eeg, hbo, hbr, eeg_labels, nirs_labels, fs=nirs_fs, mode='simple'
    )
    
    # Subsample for computational tractability
    max_samples = 3000
    if len(event_labels) > max_samples:
        print(f"\nSubsampling from {len(event_labels)} to {max_samples} samples...")
        np.random.seed(42)
        indices = np.random.choice(len(event_labels), max_samples, replace=False)
        features_sub = features_simple[indices]
        targets_sub = event_labels[indices]
    else:
        features_sub = features_simple
        targets_sub = event_labels
        indices = None
    
    print(f"\nFeatures: {feature_names_simple}")
    print(f"Target: Event labels (0=baseline, 1=target)")
    
    # Run Enhanced PIDF
    pidf = EnhancedPIDF(
        features_sub, targets_sub, 
        feature_names=feature_names_simple,
        num_iterations=150
    )
    
    results = pidf.run_full_analysis()
    pidf.print_summary()
    
    # Visualize
    viz_path = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'pidf_bimodal.png')
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    visualize_pidf_results(pidf, save_path=viz_path)
    
    # =========================================================================
    # Analysis 2: Basic Features (with power/oxygenation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Basic Features PIDF (4 features)")
    print("=" * 70)
    
    features_basic, feature_names_basic = prepare_advanced_features(
        eeg, hbo, hbr, eeg_labels, nirs_labels, fs=nirs_fs, mode='basic'
    )
    
    print(f"\nFeatures ({len(feature_names_basic)}): {feature_names_basic}")
    
    if indices is not None:
        features_basic_sub = features_basic[indices]
    else:
        features_basic_sub = features_basic
    
    pidf_basic = EnhancedPIDF(
        features_basic_sub, targets_sub,
        feature_names=feature_names_basic,
        num_iterations=120
    )
    
    results_basic = pidf_basic.run_full_analysis()
    pidf_basic.print_summary()
    
    # Visualize
    viz_path_basic = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'pidf_basic.png')
    visualize_pidf_results(pidf_basic, save_path=viz_path_basic)
    
    # =========================================================================
    # Analysis 3: All N-back Conditions Comparison (0-back, 2-back, 3-back)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Complete N-back Comparison (0-back, 2-back, 3-back)")
    print("=" * 70)
    
    load_labels = epoch_info['load_labels']
    
    # Include 0-back by checking for load_labels == 0 where there are events
    # For 0-back, we need to identify where load_labels == 0 but event_labels can be either 0 or 1
    unique_loads = [0, 2, 3]  # Explicitly include 0-back
    
    load_results = {}
    
    for load_level in unique_loads:
        mask = load_labels == load_level
        n_samples = np.sum(mask)
        
        if n_samples > 300:  # Lower threshold to include 0-back
            print(f"\n{'='*50}")
            print(f"--- {load_level}-back Task ({n_samples} samples) ---")
            print(f"{'='*50}")
            
            features_load = features_simple[mask]
            targets_load = event_labels[mask]
            
            # Check class balance
            n_target = np.sum(targets_load == 1)
            n_baseline = np.sum(targets_load == 0)
            print(f"Class distribution: target={n_target}, baseline={n_baseline}")
            
            # Subsample if needed
            if len(targets_load) > 2000:
                np.random.seed(42 + load_level)  # Different seed per condition
                idx = np.random.choice(len(targets_load), 2000, replace=False)
                features_load = features_load[idx]
                targets_load = targets_load[idx]
            
            if len(np.unique(targets_load)) > 1:  # Need both classes
                pidf_load = EnhancedPIDF(
                    features_load, targets_load,
                    feature_names=feature_names_simple,
                    num_iterations=100
                )
                
                load_results[load_level] = pidf_load.run_full_analysis()
                pidf_load.print_summary()
            else:
                print(f"  Skipping {load_level}-back: only one class present")
    
    # =========================================================================
    # Visualization: N-back Comparison
    # =========================================================================
    if load_results:
        viz_path_nback = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'pidf_nback_comparison.png')
        visualize_nback_comparison(load_results, feature_names_simple, save_path=viz_path_nback)
    
    # =========================================================================
    # Save Comprehensive Results Summary
    # =========================================================================
    results_file = os.path.join(os.path.dirname(__file__), '..', 'result', 'pidf_analysis_results.txt')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("PIDF Analysis Results for EEG/fNIRS Data\n")
        f.write("Enhanced Version with Redundancy/Interference Detection\n")
        f.write("=" * 70 + "\n\n")
        
        # Analysis 1 results
        f.write("1. Basic Bimodal Analysis (Simple Features):\n")
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(f"\n  {r.feature_name}:\n")
            f.write(f"    MI (Individual): {r.mutual_information:.4f}\n")
            f.write(f"    Synergy (θ > 0): {r.synergy:.4f}\n")
            f.write(f"    Redundancy (θ < 0): {r.redundancy:.4f}\n")
            f.write(f"    Unique Info: {r.unique_info:.4f}\n")
            f.write(f"    Net Contribution: {r.total_contribution:.4f}\n")
            f.write(f"    Interaction Type: {r.interaction_type}\n")
            if r.theta_values:
                f.write(f"    Raw θ values: {r.theta_values}\n")
        
        # Analysis 2 results
        f.write("\n\n2. Basic Features Analysis:\n")
        f.write("-" * 50 + "\n")
        for r in results_basic:
            net_theta = np.mean(list(r.theta_values.values())) if r.theta_values else 0
            f.write(f"  {r.feature_name}: MI={r.mutual_information:.4f}, theta_avg={net_theta:.4f}, type={r.interaction_type}\n")
        
        # Analysis 3 results - N-back comparison
        f.write("\n\n3. Complete N-back Comparison (0, 2, 3-back):\n")
        f.write("-" * 50 + "\n")
        f.write("\nDetailed Comparison Table:\n\n")
        
        # Header
        f.write(f"{'Condition':<12} {'Feature':<12} {'MI':>10} {'Synergy':>10} {'Redundancy':>10} {'Net θ':>10} {'Type':<15}\n")
        f.write("-" * 81 + "\n")
        
        for load_level in sorted(load_results.keys()):
            load_res = load_results[load_level]
            for r in load_res:
                net_theta = np.mean(list(r.theta_values.values())) if r.theta_values else 0
                f.write(f"{load_level}-back{'':<6} {r.feature_name:<12} {r.mutual_information:>10.4f} "
                       f"{r.synergy:>10.4f} {r.redundancy:>10.4f} {net_theta:>10.4f} {r.interaction_type:<15}\n")
            f.write("\n")
        
        # Summary statistics
        f.write("\n" + "=" * 81 + "\n")
        f.write("Summary by Condition:\n")
        f.write("-" * 50 + "\n")
        
        for load_level in sorted(load_results.keys()):
            load_res = load_results[load_level]
            total_mi = sum(r.mutual_information for r in load_res)
            total_syn = sum(r.synergy for r in load_res)
            total_red = sum(r.redundancy for r in load_res)
            net_effect = total_syn - total_red
            
            # Determine overall interaction
            if net_effect > 0.001:
                interaction = "SYNERGISTIC (EEG & fNIRS enhance each other)"
            elif net_effect < -0.001:
                interaction = "REDUNDANT/INTERFERENCE (modalities compete)"
            else:
                interaction = "INDEPENDENT (modalities provide separate info)"
            
            f.write(f"\n{load_level}-back:\n")
            f.write(f"  Total MI: {total_mi:.4f}\n")
            f.write(f"  Total Synergy: {total_syn:.4f}\n")
            f.write(f"  Total Redundancy: {total_red:.4f}\n")
            f.write(f"  Net Effect (Syn - Red): {net_effect:.4f}\n")
            f.write(f"  Interpretation: {interaction}\n")
        
        # Key findings
        f.write("\n\n" + "=" * 81 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-" * 50 + "\n")
        
        if len(load_results) >= 2:
            sorted_loads = sorted(load_results.keys())
            for i in range(len(sorted_loads) - 1):
                load1, load2 = sorted_loads[i], sorted_loads[i+1]
                res1, res2 = load_results[load1], load_results[load2]
                
                net1 = sum(r.synergy - r.redundancy for r in res1)
                net2 = sum(r.synergy - r.redundancy for r in res2)
                
                change = net2 - net1
                if change > 0.001:
                    f.write(f"\n• {load1}-back → {load2}-back: Synergy INCREASES (+{change:.4f})\n")
                    f.write(f"  → Higher cognitive load promotes multimodal integration\n")
                elif change < -0.001:
                    f.write(f"\n• {load1}-back → {load2}-back: Synergy DECREASES ({change:.4f})\n")
                    f.write(f"  → Higher cognitive load may cause modality interference\n")
                else:
                    f.write(f"\n• {load1}-back → {load2}-back: No significant change ({change:.4f})\n")
    
    print(f"\n\nResults saved to: {results_file}")
    print("\n" + "=" * 70)
    print("PIDF ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
