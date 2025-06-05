"""
European Data Loader for Transfer Prediction Model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class EuropeanDataLoader:
    """Load and process data from all 7 major European football leagues."""
    
    def __init__(self):
        """Initialize with league mappings."""
        self.european_leagues = {
            'premier_league': {'path': 'England', 'name': 'Premier-League'},
            'la_liga': {'path': 'Spain', 'name': 'La-Liga'},
            'serie_a': {'path': 'Italy', 'name': 'Serie-A'},
            'bundesliga': {'path': 'Germany', 'name': 'Bundesliga'},
            'ligue_1': {'path': 'France', 'name': 'Ligue-1'},
            'primeira_liga': {'path': 'Portugal', 'name': 'Primeira-Liga'},
            'eredivisie': {'path': 'Netherlands', 'name': 'Eredivisie'}
        }
        
        self.seasons = [
            '2017-2018', '2018-2019', '2019-2020', '2020-2021',
            '2021-2022', '2022-2023', '2023-2024'
        ]
        
        self.stat_types = ['standard', 'shooting', 'passing', 'defense', 'misc', 'gca', 'playingtime']
        
        print(f"EuropeanDataLoader initialized with {len(self.european_leagues)} leagues")
    
    def get_league_data_path(self, league: str, season: str, data_root: str = "data/raw") -> Path:
        """Get the data path for a specific league and season."""
        country_path = self.european_leagues[league]['path']
        league_name = self.european_leagues[league]['name']
        return Path(data_root) / "european_leagues" / country_path / f"{league_name}_{season}"
    
    def load_stat_file(self, league: str, season: str, stat_type: str, data_root: str = "data/raw") -> pd.DataFrame:
        """Load a specific stat file for a league/season."""
        season_path = self.get_league_data_path(league, season, data_root)
        stat_file = season_path / f"{stat_type}_stats.csv"
        
        if not stat_file.exists():
            print(f"File not found: {stat_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(stat_file)
            df['league'] = league
            df['season'] = season
            return df
        except Exception as e:
            print(f"Error loading {stat_file}: {e}")
            return pd.DataFrame()
    
    def combine_stat_types(self, league: str, season: str, data_root: str = "data/raw") -> pd.DataFrame:
        """Combine all stat types for a league/season."""
        print(f"Combining stats for {league} {season}")
        
        # Load standard stats as base
        base_df = self.load_stat_file(league, season, 'standard', data_root)
        if base_df.empty:
            return pd.DataFrame()
        
        # Merge other stat types
        for stat_type in ['shooting', 'passing', 'defense', 'misc', 'gca', 'playingtime']:
            stat_df = self.load_stat_file(league, season, stat_type, data_root)
            
            if not stat_df.empty:
                merge_cols = ['player']
                if 'player_id' in stat_df.columns:
                    merge_cols.append('player_id')
                
                cols_to_add = [col for col in stat_df.columns 
                              if col not in base_df.columns and col not in ['league', 'season']]
                
                if cols_to_add:
                    merge_df = stat_df[merge_cols + cols_to_add]
                    base_df = base_df.merge(merge_df, on=merge_cols, how='left')
        
        print(f"Combined dataset: {len(base_df)} players, {len(base_df.columns)} columns")
        return base_df
    
    def load_all_european_data(self, data_root: str = "data/raw", leagues: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data from all European leagues."""
        leagues = leagues or list(self.european_leagues.keys())
        print(f"Loading data from {len(leagues)} European leagues")
        
        all_league_data = []
        
        for league in leagues:
            print(f"\nLoading {league}...")
            league_data = []
            
            for season in self.seasons:
                df = self.combine_stat_types(league, season, data_root)
                if not df.empty:
                    league_data.append(df)
            
            if league_data:
                league_combined = pd.concat(league_data, ignore_index=True)
                all_league_data.append(league_combined)
                print(f"Success {league}: {len(league_combined)} player-seasons")
            else:
                print(f"Failed {league}: No data loaded")
        
        if not all_league_data:
            raise ValueError("No data could be loaded from any league")
        
        # Combine all leagues
        european_dataset = pd.concat(all_league_data, ignore_index=True)
        
        print(f"""
EUROPEAN DATASET LOADED SUCCESSFULLY!
Total Records: {len(european_dataset):,}
Leagues: {european_dataset['league'].nunique()}
Seasons: {european_dataset['season'].nunique()}
Columns: {len(european_dataset.columns)}
        """)
        
        return european_dataset
    
    def save_combined_dataset(self, df: pd.DataFrame, 
                             output_path: str = "data/processed/european_combined.csv"):
        """Save the combined European dataset."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        print(f"Saved combined European dataset to {output_file}")
        print(f"Dataset size: {len(df):,} rows × {len(df.columns)} columns")
        
        return output_file

if __name__ == "__main__":
    loader = EuropeanDataLoader()
    # Test with one league
    df = loader.load_stat_file('premier_league', '2023-2024', 'standard')
    print(f"Test load: {len(df)} records")
