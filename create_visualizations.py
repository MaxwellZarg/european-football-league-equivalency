"""
New Visualizations for Updated Medium Article
Creates publication-quality charts for the performance prediction system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PerformancePredictionVisualizer:
    """Create visualizations for the performance prediction system."""
    
    def __init__(self):
        """Initialize with actual results from your system."""
        
        # Your actual equivalency factors from the pipeline output
        self.equivalency_factors = {
            'ligue_1_to_premier_league': 0.815,
            'la_liga_to_premier_league': 0.813,
            'ligue_1_to_serie_a': 0.805,
            'ligue_1_to_la_liga': 0.934,
            'bundesliga_to_premier_league': 0.385,
            'premier_league_to_serie_a': 1.012,
            'serie_a_to_premier_league': 0.593,
            'premier_league_to_la_liga': 0.764,
            'ligue_1_to_bundesliga': 0.906,
            'primeira_liga_to_ligue_1': 0.682
        }
        
        # Sample sizes from your output
        self.sample_sizes = {
            'ligue_1_to_premier_league': 65,
            'la_liga_to_premier_league': 60,
            'ligue_1_to_serie_a': 52,
            'ligue_1_to_la_liga': 44,
            'bundesliga_to_premier_league': 43,
            'premier_league_to_serie_a': 37,
            'serie_a_to_premier_league': 36,
            'premier_league_to_la_liga': 33,
            'ligue_1_to_bundesliga': 32,
            'primeira_liga_to_ligue_1': 32
        }
        
        # League colors
        self.colors = {
            'premier_league': '#37003C',  # Purple
            'la_liga': '#FF6B35',         # Orange
            'serie_a': '#0066CC',         # Blue
            'bundesliga': '#D70027',      # Red
            'ligue_1': '#1E3A8A',        # Dark Blue
            'primeira_liga': '#00A86B',   # Green
            'eredivisie': '#FF6B00'       # Dutch Orange
        }
    
    def create_equivalency_matrix_heatmap(self, save_path=None):
        """Create a heatmap showing equivalency factors between leagues."""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # League names for display
        leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Primeira Liga', 'Eredivisie']
        league_codes = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1', 'primeira_liga', 'eredivisie']
        
        # Create matrix
        matrix = np.full((len(leagues), len(leagues)), np.nan)
        
        # Fill in known factors
        factor_mapping = {
            ('ligue_1', 'premier_league'): 0.815,
            ('la_liga', 'premier_league'): 0.813,
            ('ligue_1', 'serie_a'): 0.805,
            ('ligue_1', 'la_liga'): 0.934,
            ('bundesliga', 'premier_league'): 0.385,
            ('premier_league', 'serie_a'): 1.012,
            ('serie_a', 'premier_league'): 0.593,
            ('premier_league', 'la_liga'): 0.764,
            ('ligue_1', 'bundesliga'): 0.906,
            ('primeira_liga', 'ligue_1'): 0.682
        }
        
        for (from_league, to_league), factor in factor_mapping.items():
            from_idx = league_codes.index(from_league)
            to_idx = league_codes.index(to_league)
            matrix[from_idx, to_idx] = factor
        
        # Create mask for diagonal (same league)
        mask = np.zeros_like(matrix, dtype=bool)
        np.fill_diagonal(mask, True)
        
        # Create heatmap
        sns.heatmap(matrix, 
                   xticklabels=leagues,
                   yticklabels=leagues,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0.8,
                   vmin=0.3,
                   vmax=1.1,
                   mask=mask,
                   cbar_kws={'label': 'Performance Translation Factor'},
                   ax=ax)
        
        ax.set_title('League Equivalency Matrix\nHow Player Performance Translates Between Leagues', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Target League', fontsize=12, fontweight='bold')
        ax.set_ylabel('Source League', fontsize=12, fontweight='bold')
        
        # Add explanation text
        ax.text(0.02, 0.98, 
               'Reading the matrix:\n• Values >1.0 = Improvement expected\n• Values <1.0 = Performance decline expected\n• Empty cells = Insufficient data',
               transform=ax.transAxes,
               fontsize=10,
               va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_model_performance_comparison(self, save_path=None):
        """Compare realistic vs. unrealistic model performance."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Unrealistic (old approach)
        ax1.bar(['Transfer Detection\n(Old Approach)'], [100], 
               color='red', alpha=0.7, width=0.5)
        ax1.set_ylim(0, 110)
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('❌ Unrealistic Results\n(Transfer Detection)', 
                     fontsize=14, fontweight='bold', color='red')
        ax1.text(0, 105, '100%\n(Red Flag!)', ha='center', va='bottom', 
                fontweight='bold', fontsize=12, color='red')
        
        # Add warning annotation
        ax1.annotate('⚠️ Perfect accuracy indicates\noverfitting or data leakage',
                    xy=(0, 100), xytext=(0.3, 80),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Right plot: Realistic (new approach)
        models = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'Ensemble']
        mae_scores = [0.0962, 0.0920, 0.0945, 0.0925]  # Convert to percentage-like scale
        accuracy_pct = [(1 - mae) * 100 for mae in mae_scores]
        
        bars = ax2.bar(models, accuracy_pct, 
                      color=['skyblue', 'green', 'orange', 'purple'], 
                      alpha=0.8)
        
        ax2.set_ylim(85, 95)
        ax2.set_ylabel('Approximate Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('✅ Realistic Results\n(Performance Prediction)', 
                     fontsize=14, fontweight='bold', color='green')
        
        # Add value labels
        for bar, acc, mae in zip(bars, accuracy_pct, mae_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{acc:.1f}%\n(MAE: {mae:.3f})', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add success annotation
        ax2.annotate('✅ Realistic accuracy reflects\nreal-world uncertainty',
                    xy=(1, 92), xytext=(2.5, 88),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_difficulty_ranking_chart(self, save_path=None):
        """Create a chart showing league transition difficulty rankings."""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for visualization
        transitions = [
            ('Bundesliga → Premier League', 0.385, 43, 'Very Hard'),
            ('Serie A → Premier League', 0.593, 36, 'Hard'),
            ('Primeira Liga → Ligue 1', 0.682, 32, 'Moderate-Hard'),
            ('Premier League → La Liga', 0.764, 33, 'Moderate'),
            ('Ligue 1 → Serie A', 0.805, 52, 'Moderate'),
            ('La Liga → Premier League', 0.813, 60, 'Moderate'),
            ('Ligue 1 → Premier League', 0.815, 65, 'Moderate'),
            ('Ligue 1 → Bundesliga', 0.906, 32, 'Easy'),
            ('Ligue 1 → La Liga', 0.934, 44, 'Easy'),
            ('Premier League → Serie A', 1.012, 37, 'Improvement')
        ]
        
        # Sort by difficulty (factor value)
        transitions.sort(key=lambda x: x[1])
        
        transition_names = [t[0] for t in transitions]
        factors = [t[1] for t in transitions]
        samples = [t[2] for t in transitions]
        difficulties = [t[3] for t in transitions]
        
        # Color coding by difficulty
        difficulty_colors = {
            'Very Hard': '#8B0000',      # Dark Red
            'Hard': '#FF4500',           # Orange Red
            'Moderate-Hard': '#FF8C00',  # Dark Orange
            'Moderate': '#FFD700',       # Gold
            'Easy': '#90EE90',           # Light Green
            'Improvement': '#006400'     # Dark Green
        }
        
        colors = [difficulty_colors[d] for d in difficulties]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(transition_names)), factors, color=colors, alpha=0.8)
        
        # Add sample size labels
        for i, (bar, factor, sample) in enumerate(zip(bars, factors, samples)):
            ax.text(factor + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{factor:.3f}\n(n={sample})', 
                   va='center', ha='left', fontweight='bold', fontsize=9)
        
        # Customize chart
        ax.set_yticks(range(len(transition_names)))
        ax.set_yticklabels(transition_names, fontsize=11)
        ax.set_xlabel('Performance Translation Factor', fontsize=12, fontweight='bold')
        ax.set_title('League Transition Difficulty Rankings\nBased on Actual Player Performance Data', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add reference lines
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(1.02, len(transitions)-1, 'Equal Performance\n(1.0)', 
               rotation=90, va='top', ha='left', fontsize=10, fontweight='bold')
        
        # Add legend
        legend_elements = [mpatches.Patch(color=color, label=difficulty) 
                          for difficulty, color in difficulty_colors.items()]
        ax.legend(handles=legend_elements, loc='lower right', 
                 title='Transition Difficulty', title_fontsize=11, fontsize=10)
        
        # Add insights box
        insights = (
            "Key Insights:\n"
            "• Bundesliga → PL is hardest transition\n"
            "• Ligue 1 players adapt well across leagues\n"
            "• PL → Serie A often improves performance\n"
            "• Sample size affects reliability"
        )
        ax.text(0.02, 0.98, insights, transform=ax.transAxes, 
               fontsize=11, va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0.3, 1.2)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_player_case_studies(self, save_path=None):
        """Create visualization showing real player case studies."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Case studies with actual predictions
        case_studies = [
            {
                'name': 'Harry Kane',
                'from': 'Bundesliga', 
                'current_stats': {'goals': 36, 'assists': 8, 'games': 32},
                'predictions': {'premier_league': 0.59, 'ligue_1': 0.47, 'serie_a': 0.55},
                'ax': ax1
            },
            {
                'name': 'Cody Gakpo\n(Historical)',
                'from': 'Eredivisie',
                'current_stats': {'goals': 9, 'assists': 12, 'games': 24},
                'predictions': {'premier_league': 0.64, 'la_liga': 0.68, 'serie_a': 0.66},
                'actual': {'premier_league': 0.64},
                'ax': ax2
            },
            {
                'name': 'Viktor Gyökeres',
                'from': 'Primeira Liga',
                'current_stats': {'goals': 29, 'assists': 10, 'games': 35},
                'predictions': {'premier_league': 0.85, 'la_liga': 0.92, 'bundesliga': 0.88},
                'ax': ax3
            },
            {
                'name': 'Model Accuracy\nValidation',
                'from': 'Various',
                'current_stats': {'goals': 0, 'assists': 0, 'games': 0},
                'predictions': {'mae': 0.092, 'r2_goals': 0.517, 'r2_assists': 0.207},
                'ax': ax4
            }
        ]
        
        # Plot first three case studies
        for i, case in enumerate(case_studies[:3]):
            ax = case['ax']
            
            # Current performance
            current_90 = (case['current_stats']['goals'] + case['current_stats']['assists']) / (case['current_stats']['games'] * 0.8)  # Approximate 90s
            
            leagues = list(case['predictions'].keys())
            predictions = list(case['predictions'].values())
            
            bars = ax.bar(leagues, predictions, alpha=0.7, 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(leagues)])
            
            # Add actual performance if available
            if 'actual' in case:
                for league, actual in case['actual'].items():
                    idx = leagues.index(league)
                    ax.bar(league, actual, alpha=0.3, color='red', 
                          label='Actual Performance')
                    ax.text(idx, actual + 0.02, f'Actual: {actual:.2f}', 
                           ha='center', va='bottom', fontweight='bold', color='red')
            
            ax.set_title(f'{case["name"]}\n({case["from"]})', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Contributions per 90', fontsize=11)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, pred in zip(bars, predictions):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{pred:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Add current stats
            current_text = f"Current: {case['current_stats']['goals']}G, {case['current_stats']['assists']}A in {case['current_stats']['games']} games"
            ax.text(0.02, 0.98, current_text, transform=ax.transAxes, 
                   fontsize=9, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
        
        # Fourth subplot: Model validation metrics
        ax4 = case_studies[3]['ax']
        
        metrics = ['Overall MAE', 'Goals R²', 'Assists R²']
        values = [0.092, 0.517, 0.207]
        colors_val = ['red', 'green', 'blue']
        
        bars = ax4.bar(metrics, values, color=colors_val, alpha=0.7)
        
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('Model Performance\nMetrics', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Score', fontsize=11)
        ax4.set_ylim(0, 0.6)
        
        # Add interpretation
        interpretation = (
            "Interpretation:\n"
            "• MAE: 0.092 = realistic accuracy\n"
            "• Goals more predictable than assists\n"
            "• Accounts for real-world uncertainty"
        )
        ax4.text(0.02, 0.98, interpretation, transform=ax4.transAxes, 
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_methodology_comparison(self, save_path=None):
        """Create a comparison showing old vs new methodology."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Old methodology (Transfer Detection)
        ax1.text(0.5, 0.9, 'OLD APPROACH', ha='center', va='center', 
                fontsize=18, fontweight='bold', color='red',
                transform=ax1.transAxes)
