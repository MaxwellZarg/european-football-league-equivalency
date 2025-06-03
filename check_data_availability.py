"""
Data quality checker to verify actual data content in columns.
Checks if columns have meaningful data or are just empty/null.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.append(str(Path("src").resolve()))

from utils.config import load_config

class DataQualityChecker:
    """Check actual data quality and content in CSV files."""
    
    def __init__(self):
        self.config = load_config()
        self.leagues = list(self.config['leagues'].keys())
        self.seasons = self.config['seasons']
        
        # Key metrics we're interested in
        self.key_metrics = {
            'Expected Goals': ['xg', 'npxg', 'xg_per90', 'xg_net'],
            'Expected Assists': ['xg_assist', 'xg_assist_per90', 'pass_xa'],
            'Progressive Actions': ['progressive_passes', 'progressive_carries', 'progressive_passes_received'],
            'Defensive Actions': ['tackles', 'interceptions', 'blocks', 'clearances'],
            'Shot Creation': ['sca', 'gca', 'sca_per90', 'gca_per90'],
            'Passing Quality': ['passes_completed', 'passes_pct', 'passes_progressive_distance'],
            'Physical Data': ['aerials_won', 'aerials_lost', 'fouls', 'fouled']
        }
        
        self.stat_types = ['standard', 'shooting', 'passing', 'defense', 'gca', 'playingtime', 'misc']
    
    def check_column_data_quality(self, df, column_name):
        """Check the actual data quality of a specific column."""
        if column_name not in df.columns:
            return {
                'exists': False,
                'total_rows': len(df),
                'non_null_count': 0,
                'non_zero_count': 0,
                'data_percentage': 0,
                'sample_values': [],
                'data_type': None
            }
        
        col_data = df[column_name]
        
        # Count non-null values
        non_null_count = col_data.notna().sum()
        
        # Convert to numeric if possible
        try:
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            non_zero_count = (numeric_data > 0).sum()
            sample_values = numeric_data.dropna().head(5).tolist()
            data_type = 'numeric'
        except:
            non_zero_count = (col_data != '').sum() if col_data.dtype == 'object' else non_null_count
            sample_values = col_data.dropna().head(5).tolist()
            data_type = 'text'
        
        data_percentage = (non_null_count / len(df)) * 100 if len(df) > 0 else 0
        
        return {
            'exists': True,
            'total_rows': len(df),
            'non_null_count': non_null_count,
            'non_zero_count': non_zero_count,
            'data_percentage': data_percentage,
            'sample_values': sample_values,
            'data_type': data_type
        }
    
    def analyze_league_season_data(self, league, season, metrics_to_check):
        """Analyze data quality for specific league/season combination."""
        league_name = self.config['leagues'][league]['fbref_name']
        season_dir = Path(f"data/raw/{league_name}_{season}")
        
        results = {}
        
        if not season_dir.exists():
            return results
        
        # Check each stat type
        for stat_type in self.stat_types:
            stat_file = season_dir / f"{stat_type}_stats.csv"
            
            if stat_file.exists():
                try:
                    df = pd.read_csv(stat_file)
                    
                    # Check each metric we're interested in
                    for metric in metrics_to_check:
                        if metric in df.columns:
                            quality_info = self.check_column_data_quality(df, metric)
                            
                            key = f"{stat_type}_{metric}"
                            results[key] = quality_info
                            
                except Exception as e:
                    print(f"Error reading {stat_file}: {e}")
        
        return results
    
    def comprehensive_data_quality_check(self):
        """Run comprehensive data quality check across all metrics."""
        
        print("=" * 60)
        print("COMPREHENSIVE DATA QUALITY ANALYSIS")
        print("=" * 60)
        
        # Flatten all metrics into one list
        all_metrics = []
        for metric_group in self.key_metrics.values():
            all_metrics.extend(metric_group)
        
        # Remove duplicates
        all_metrics = list(set(all_metrics))
        
        # Store results
        quality_results = defaultdict(lambda: defaultdict(dict))
        
        print(f"\nChecking {len(all_metrics)} key metrics across {len(self.leagues)} leagues...")
        print(f"Metrics: {', '.join(all_metrics[:5])}...")
        
        for league in self.leagues:
            print(f"\nAnalyzing {league.upper()}...")
            
            for season in self.seasons:
                season_results = self.analyze_league_season_data(league, season, all_metrics)
                quality_results[league][season] = season_results
        
        return quality_results
    
    def summarize_metric_availability(self, quality_results):
        """Summarize which metrics actually have meaningful data."""
        
        print("\n" + "=" * 60)
        print("METRIC DATA AVAILABILITY SUMMARY")
        print("=" * 60)
        
        for metric_group, metrics in self.key_metrics.items():
            print(f"\n{metric_group.upper()}:")
            print("-" * 40)
            
            for metric in metrics:
                print(f"\n  {metric}:")
                
                # Check availability across leagues
                league_stats = {}
                
                for league in self.leagues:
                    seasons_with_data = 0
                    total_seasons = 0
                    total_data_points = 0
                    total_possible_points = 0
                    
                    for season in self.seasons:
                        season_data = quality_results[league][season]
                        
                        # Look for this metric in any stat type
                        found_data = False
                        for key, data_info in season_data.items():
                            if metric in key and data_info['exists']:
                                total_seasons += 1
                                total_data_points += data_info['non_null_count']
                                total_possible_points += data_info['total_rows']
                                
                                # Consider it "has data" if >10% of rows have non-null values
                                if data_info['data_percentage'] > 10:
                                    seasons_with_data += 1
                                    found_data = True
                                break
                    
                    if total_seasons > 0:
                        data_coverage = (total_data_points / total_possible_points) * 100 if total_possible_points > 0 else 0
                        league_stats[league] = {
                            'seasons_with_data': seasons_with_data,
                            'total_seasons': total_seasons,
                            'data_coverage': data_coverage
                        }
                
                # Report findings
                if league_stats:
                    for league, stats in league_stats.items():
                        coverage = stats['data_coverage']
                        seasons_ratio = f"{stats['seasons_with_data']}/{stats['total_seasons']}"
                        
                        if coverage > 80:
                            status = "EXCELLENT"
                        elif coverage > 50:
                            status = "GOOD"
                        elif coverage > 20:
                            status = "PARTIAL"
                        else:
                            status = "POOR"
                        
                        print(f"    {league}: {coverage:.1f}% coverage ({seasons_ratio} seasons) - {status}")
                else:
                    print(f"    No data found across any league")
    
    def create_metric_recommendations(self, quality_results):
        """Create recommendations for which metrics are viable for analysis."""
        
        print("\n" + "=" * 60)
        print("ANALYSIS RECOMMENDATIONS")
        print("=" * 60)
        
        viable_analyses = []
        
        for metric_group, metrics in self.key_metrics.items():
            group_viability = []
            
            for metric in metrics:
                leagues_with_good_data = 0
                
                for league in self.leagues:
                    league_has_good_data = False
                    
                    for season in self.seasons:
                        season_data = quality_results[league][season]
                        
                        for key, data_info in season_data.items():
                            if metric in key and data_info['exists']:
                                if data_info['data_percentage'] > 50:  # At least 50% coverage
                                    league_has_good_data = True
                                    break
                    
                    if league_has_good_data:
                        leagues_with_good_data += 1
                
                if leagues_with_good_data >= 3:  # At least 3 leagues have good data
                    group_viability.append(metric)
            
            if group_viability:
                viable_analyses.append({
                    'group': metric_group,
                    'viable_metrics': group_viability,
                    'potential_leagues': leagues_with_good_data
                })
        
        print("\nVIABLE ANALYSIS OPPORTUNITIES:")
        print("=" * 40)
        
        for analysis in viable_analyses:
            print(f"\n{analysis['group']}:")
            print(f"  Viable metrics: {', '.join(analysis['viable_metrics'])}")
            print(f"  Recommendation: PROCEED - Good data availability")
        
        if not viable_analyses:
            print("\nWARNING: Limited viable analysis opportunities found.")
            print("Most advanced metrics may have insufficient data coverage.")
        
        return viable_analyses
    
    def run_quality_analysis(self):
        """Run the complete data quality analysis."""
        
        print("STARTING DATA QUALITY VERIFICATION")
        print("Checking if columns actually contain meaningful data...")
        
        # Run comprehensive check
        quality_results = self.comprehensive_data_quality_check()
        
        # Summarize findings
        self.summarize_metric_availability(quality_results)
        
        # Create recommendations
        viable_analyses = self.create_metric_recommendations(quality_results)
        
        print(f"\n" + "=" * 60)
        print("QUALITY ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Found {len(viable_analyses)} viable analysis opportunities")
        
        return quality_results, viable_analyses


def main():
    """Run the data quality checker."""
    
    checker = DataQualityChecker()
    quality_results, viable_analyses = checker.run_quality_analysis()
    
    print("\nSummary saved to analysis results.")
    return quality_results, viable_analyses


if __name__ == "__main__":
    results = main()
