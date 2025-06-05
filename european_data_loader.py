"""
European Data Loader for Transfer Prediction Model
Extends the existing DataLoader to handle all 7 European leagues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EuropeanDataLoader:
    """Load and process data from all 7 major European football leagues."""
    
    def __init__(self, config_path: str = "config/league_mappings.yaml"):
        """Initialize with configuration file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # League mappings for European leagues (all in european_leagues directory)
        self.european_leagues = {
            'premier_league': {'path': 'England', 'name': 'Premier-League'},
            'la_liga': {'path': 'Spain', 'name': 'La-Liga'},
            'serie_a': {'path': 'Italy', 'name': 'Serie-A'},
            'bundesliga': {'path': 'Germany', 'name': 'Bundesliga'},
            'ligue_1': {'path': 'France', 'name': 'Ligue-1'},
            'primeira_liga': {'path': 'Portugal', 'name': 'Primeira-Liga'},
            'eredivisie': {'path': 'Netherlands', 'name': 'Eredivisie'}
        }
        
        # Available seasons (complete stats era)
        self.seasons = [
            '2017-2018', '2018-2019', '2019-2020', '2020-2021',
            '2021-2022', '2022-2023', '2023-2024'
        ]
        
        # Stat file types available
        self.stat_types = [
            'standard', 'shooting', 'passing', 'defense', 
            'misc', 'gca', 'playingtime'
        ]
        
        logger.info(f"EuropeanDataLoader initialized with {len(self.european_leagues)} leagues")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return {}
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_league_data_path(self, league: str, season: str, data_root: str = "data/raw") -> Path:
        """Get the data path for a specific league and season."""
        if league in self.european_leagues:
            country_path = self.european_leagues[league]['path']
            league_name = self.european_leagues[league]['name']
            
            # All leagues now in european_leagues structure
            return Path(data_root) / "european_leagues" / country_path / f"{league_name}_{season}"
        else:
            raise ValueError(f"Unknown league: {league}")
    
    def load_stat_file(self, league: str, season: str, stat_type: str, 
                      data_root: str = "data/raw") -> pd.DataFrame:
        """Load a specific stat file for a league/season."""
        
        season_path = self.get_league_data_path(league, season, data_root)
        stat_file = season_path / f"{stat_type}_stats.csv"
        
        if not stat_file.exists():
            logger.warning(f"File not found: {stat_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(stat_file)
            
            # Add metadata
            df['league'] = league
            df['season'] = season
            df['stat_type'] = stat_type
            
            logger.debug(f"Loaded {len(df)} rows from {stat_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {stat_file}: {e}")
            return pd.DataFrame()
    
    def combine_stat_types(self, league: str, season: str, 
                          data_root: str = "data/raw") -> pd.DataFrame:
        """Combine all stat types for a league/season into one comprehensive dataset."""
        
        logger.info(f"Combining stats for {league} {season}")
        
        # Load standard stats as base (most complete)
        base_df = self.load_stat_file(league, season, 'standard', data_root)
        
        if base_df.empty:
            logger.warning(f"No standard stats found for {league} {season}")
            return pd.DataFrame()
        
        # Core columns for merging
        merge_cols = ['player']
        if 'player_id' in base_df.columns:
            merge_cols.append('player_id')
        
        # Add other stat types
        for stat_type in ['shooting', 'passing', 'defense', 'misc', 'gca', 'playingtime']:
            stat_df = self.load_stat_file(league, season, stat_type, data_root)
            
            if not stat_df.empty:
                # Remove duplicate columns before merging
                cols_to_add = [col for col in stat_df.columns 
                              if col not in base_df.columns and col not in ['league', 'season', 'stat_type']]
                
                if cols_to_add:
                    merge_df = stat_df[merge_cols + cols_to_add].copy()
                    
                    # Merge on player (and player_id if available)
                    base_df = base_df.merge(merge_df, on=merge_cols, how='left', suffixes=('', f'_{stat_type}'))
                    
                    logger.debug(f"Added {len(cols_to_add)} columns from {stat_type}")
        
        logger.info(f"Combined dataset: {len(base_df)} players, {len(base_df.columns)} columns")
        return base_df
    
    def clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the combined dataset."""
        
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Standardize player names
        if 'player' in df_clean.columns:
            df_clean['player'] = df_clean['player'].str.strip()
            df_clean = df_clean[df_clean['player'].notna()]
            df_clean = df_clean[df_clean['player'] != 'Player']
        
        # Convert numeric columns
        numeric_columns = [
            'age', 'games', 'games_starts', 'minutes', 'minutes_90s',
            'goals', 'assists', 'goals_assists', 'shots', 'shots_on_target',
            'passes_completed', 'passes', 'tackles', 'interceptions',
            'cards_yellow', 'cards_red', 'xg', 'npxg', 'xg_assist'
        ]
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Create derived metrics
        if 'minutes' in df_clean.columns and 'games' in df_clean.columns:
            df_clean['minutes_per_game'] = df_clean['minutes'] / df_clean['games'].replace(0, np.nan)
        
        if 'goals' in df_clean.columns and 'assists' in df_clean.columns:
            df_clean['goals_assists'] = df_clean['goals'].fillna(0) + df_clean['assists'].fillna(0)
        
        # Filter out players with minimal playing time
        min_minutes = 90  # At least 1 full game equivalent
        if 'minutes' in df_clean.columns:
            df_clean = df_clean[df_clean['minutes'] >= min_minutes]
        
        logger.info(f"Cleaned dataset: {len(df_clean)} players after filtering")
        return df_clean
    
    def load_league_all_seasons(self, league: str, data_root: str = "data/raw") -> pd.DataFrame:
        """Load all seasons for a specific league."""
        
        logger.info(f"Loading all seasons for {league}")
        all_seasons = []
        
        for season in self.seasons:
            df = self.combine_stat_types(league, season, data_root)
            if not df.empty:
                df_clean = self.clean_and_standardize(df)
                if not df_clean.empty:
                    all_seasons.append(df_clean)
        
        if not all_seasons:
            logger.warning(f"No data loaded for {league}")
            return pd.DataFrame()
        
        combined = pd.concat(all_seasons, ignore_index=True)
        logger.info(f"Loaded {league}: {len(combined)} player-seasons across {len(all_seasons)} seasons")
        
        return combined
    
    def load_all_european_data(self, data_root: str = "data/raw", 
                              leagues: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data from all European leagues."""
        
        leagues = leagues or list(self.european_leagues.keys())
        logger.info(f"Loading data from {len(leagues)} European leagues")
        
        all_league_data = []
        league_summary = {}
        
        for league in leagues:
            league_df = self.load_league_all_seasons(league, data_root)
            
            if not league_df.empty:
                all_league_data.append(league_df)
                league_summary[league] = len(league_df)
                logger.info(f"Success {league}: {len(league_df)} player-seasons")
            else:
                league_summary[league] = 0
                logger.warning(f"Failed {league}: No data loaded")
        
        if not all_league_data:
            raise ValueError("No data could be loaded from any league")
        
        # Combine all leagues
        european_dataset = pd.concat(all_league_data, ignore_index=True)
        
        # Add some derived features
        european_dataset = self._add_derived_features(european_dataset)
        
        # Summary statistics
        total_players = len(european_dataset)
        total_seasons = european_dataset['season'].nunique()
        total_leagues = european_dataset['league'].nunique()
        
        logger.info(f"""
        EUROPEAN DATASET LOADED SUCCESSFULLY!
        =======================================
        Total Records: {total_players:,}
        Leagues: {total_leagues}
        Seasons: {total_seasons}
        Columns: {len(european_dataset.columns)}
        
        League Breakdown:
        {chr(10).join([f"  {league}: {count:,}" for league, count in league_summary.items()])}
        """)
        
        return european_dataset
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for analysis."""
        
        df_enhanced = df.copy()
        
        # Performance per 90 minutes
        if 'minutes_90s' in df.columns and df['minutes_90s'].notna().sum() > 0:
            for stat in ['goals', 'assists', 'shots', 'passes_completed', 'tackles']:
                if stat in df.columns:
                    df_enhanced[f'{stat}_per90'] = (
                        df[stat] / df['minutes_90s'].replace(0, np.nan)
                    )
        
        # Efficiency metrics
        if 'shots' in df.columns and 'goals' in df.columns:
            df_enhanced['shot_conversion'] = df['goals'] / df['shots'].replace(0, np.nan)
        
        if 'passes' in df.columns and 'passes_completed' in df.columns:
            df_enhanced['pass_accuracy'] = df['passes_completed'] / df['passes'].replace(0, np.nan)
        
        # Age groups for analysis
        if 'age' in df.columns:
            df_enhanced['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 21, 25, 29, 35, 50], 
                labels=['U21', '21-25', '26-29', '30-35', '35+'],
                include_lowest=True
            )
        
        # Playing time categories
        if 'minutes' in df.columns:
            df_enhanced['playing_time_category'] = pd.cut(
                df['minutes'],
                bins=[0, 500, 1500, 2500, 5000],
                labels=['Minimal', 'Rotational', 'Regular', 'Key Player'],
                include_lowest=True
            )
        
        logger.debug("Added derived features to dataset")
        return df_enhanced
    
    def save_combined_dataset(self, df: pd.DataFrame, 
                             output_path: str = "data/processed/european_combined.csv"):
        """Save the combined European dataset."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved combined European dataset to {output_file}")
        logger.info(f"Dataset size: {len(df):,} rows × {len(df.columns)} columns")
        
        return output_file
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive summary of the dataset."""
        
        summary = {
            'total_records': len(df),
            'total_players': df['player'].nunique() if 'player' in df.columns else 0,
            'leagues': df['league'].value_counts().to_dict() if 'league' in df.columns else {},
            'seasons': df['season'].value_counts().to_dict() if 'season' in df.columns else {},
            'columns': list(df.columns),
            'missing_data': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'date_range': {
                'first_season': df['season'].min() if 'season' in df.columns else None,
                'last_season': df['season'].max() if 'season' in df.columns else None
            }
        }
        
        return summary


def main():
    """Test the European Data Loader"""
    
    print("Testing European Data Loader")
    print("=" * 50)
    
    loader = EuropeanDataLoader()
    
    # Test loading a single league
    print("\nTesting single league load...")
