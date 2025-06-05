"""
Feature Engineering Pipeline for Transfer Prediction
Creates advanced features for machine learning models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for transfer prediction."""
    
    def __init__(self, include_advanced_features: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            include_advanced_features: Whether to include computationally expensive features
        """
        self.include_advanced = include_advanced_features
        self.scalers = {}
        self.encoders = {}
        
        print(f"FeatureEngineer initialized (advanced_features={include_advanced_features})")
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Labeled dataset with transfer information
            
        Returns:
            DataFrame with engineered features
        """
        print("Starting comprehensive feature engineering...")
        
        # Create copy to avoid modifying original
        engineered_df = df.copy()
        
        # Apply feature engineering steps
        engineered_df = self._engineer_performance_features(engineered_df)
        engineered_df = self._engineer_comparative_features(engineered_df)
        engineered_df = self._engineer_contextual_features(engineered_df)
        engineered_df = self._engineer_transfer_destination_features(engineered_df)
        
        # Clean and validate features
        engineered_df = self._clean_features(engineered_df)
        
        print(f"Feature engineering complete: {len(engineered_df.columns)} total features")
        return engineered_df
    
    def _engineer_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer performance-based features."""
        
        print("Engineering performance features...")
        
        # Basic rate statistics per 90 minutes
        if 'minutes' in df.columns:
            df['minutes_90s'] = df['minutes'] / 90
            
            for stat in ['goals', 'assists', 'shots', 'tackles']:
                if stat in df.columns:
                    df[f'{stat}_per90'] = df[stat] / df['minutes_90s'].replace(0, np.nan)
        
        # Efficiency metrics
        if 'shots' in df.columns and 'goals' in df.columns:
            df['shot_efficiency'] = df['goals'] / df['shots'].replace(0, np.nan)
        
        if 'passes' in df.columns and 'passes_completed' in df.columns:
            df['pass_completion_rate'] = df['passes_completed'] / df['passes'].replace(0, np.nan)
        
        # Composite performance score
        if 'goals' in df.columns and 'assists' in df.columns:
            df['offensive_contribution'] = df['goals'].fillna(0) + df['assists'].fillna(0)
            
            if 'minutes_90s' in df.columns:
                df['offensive_contribution_per90'] = (
                    df['offensive_contribution'] / df['minutes_90s'].replace(0, np.nan)
                )
        
        # Expected vs actual performance
        if 'xg' in df.columns and 'goals' in df.columns:
            df['goals_above_expected'] = df['goals'] - df['xg']
        
        return df
    
    def _engineer_comparative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comparative features (vs league, position, peers)."""
        
        print("Engineering comparative features...")
        
        # League percentiles for key metrics
        for metric in ['goals', 'assists', 'minutes', 'offensive_contribution']:
            if metric in df.columns:
                df[f'{metric}_league_pct'] = df.groupby(['league', 'season'])[metric].rank(pct=True)
        
        # Age-adjusted performance
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['is_peak_age'] = ((df['age'] >= 25) & (df['age'] <= 29)).astype(int)
            df['is_young_talent'] = (df['age'] <= 23).astype(int)
            df['is_veteran'] = (df['age'] >= 32).astype(int)
        
        # Team context comparisons
        if 'team' in df.columns:
            team_metrics = df.groupby(['team', 'season']).agg({
                'goals': ['sum', 'mean'],
                'assists': ['sum', 'mean'],
                'minutes': 'sum'
            })
            
            team_metrics.columns = ['team_total_goals', 'team_avg_goals', 'team_total_assists', 
                                   'team_avg_assists', 'team_total_minutes']
            team_metrics = team_metrics.reset_index()
            
            df = df.merge(team_metrics, on=['team', 'season'], how='left')
            
            # Player's share of team performance
            if 'team_total_goals' in df.columns:
                df['goals_team_share'] = df['goals'] / df['team_total_goals'].replace(0, np.nan)
                df['assists_team_share'] = df['assists'] / df['team_total_assists'].replace(0, np.nan)
        
        return df
    
    def _engineer_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer contextual features (league, club, market)."""
        
        print("Engineering contextual features...")
        
        # League characteristics (no hierarchy - all equal)
        league_characteristics = {
            'premier_league': {'style': 'physical', 'pace': 'high', 'technical': 'medium'},
            'la_liga': {'style': 'technical', 'pace': 'medium', 'technical': 'high'},
            'bundesliga': {'style': 'tactical', 'pace': 'high', 'technical': 'high'},
            'serie_a': {'style': 'tactical', 'pace': 'medium', 'technical': 'high'},
            'ligue_1': {'style': 'physical', 'pace': 'medium', 'technical': 'medium'},
            'primeira_liga': {'style': 'technical', 'pace': 'medium', 'technical': 'medium'},
            'eredivisie': {'style': 'attacking', 'pace': 'high', 'technical': 'high'}
        }
        
        # Create league style features (no prestige ranking)
        style_mapping = {'physical': 1, 'technical': 2, 'tactical': 3, 'attacking': 4}
        pace_mapping = {'medium': 1, 'high': 2}
        technical_mapping = {'medium': 1, 'high': 2}
        
        df['league_style'] = df['league'].map(lambda x: style_mapping.get(league_characteristics.get(x, {}).get('style'), 0))
        df['league_pace'] = df['league'].map(lambda x: pace_mapping.get(league_characteristics.get(x, {}).get('pace'), 0))
        df['league_technical'] = df['league'].map(lambda x: technical_mapping.get(league_characteristics.get(x, {}).get('technical'), 0))
        
        # Position-specific features
        if 'position' in df.columns:
            # Simplified position grouping
            position_mapping = {
                'GK': 'goalkeeper',
                'DF': 'defender', 'CB': 'defender', 'LB': 'defender', 'RB': 'defender',
                'MF': 'midfielder', 'CM': 'midfielder', 'CAM': 'midfielder', 'CDM': 'midfielder',
                'FW': 'forward', 'CF': 'forward', 'LW': 'forward', 'RW': 'forward'
            }
            
            df['position_group'] = df['position'].map(position_mapping).fillna('unknown')
        
        # Season progression
        season_mapping = {
            '2017-2018': 1, '2018-2019': 2, '2019-2020': 3, '2020-2021': 4,
            '2021-2022': 5, '2022-2023': 6, '2023-2024': 7
        }
        df['season_number'] = df['season'].map(season_mapping)
        
        return df
    
    def _engineer_transfer_destination_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for transfer destination prediction."""
        
        print("Engineering transfer destination features...")
        
        # League characteristics for reference
        league_characteristics = {
            'premier_league': {'style': 'physical', 'pace': 'high', 'technical': 'medium'},
            'la_liga': {'style': 'technical', 'pace': 'medium', 'technical': 'high'},
            'bundesliga': {'style': 'tactical', 'pace': 'high', 'technical': 'high'},
            'serie_a': {'style': 'tactical', 'pace': 'medium', 'technical': 'high'},
            'ligue_1': {'style': 'physical', 'pace': 'medium', 'technical': 'medium'},
            'primeira_liga': {'style': 'technical', 'pace': 'medium', 'technical': 'medium'},
            'eredivisie': {'style': 'attacking', 'pace': 'high', 'technical': 'high'}
        }
        
        style_mapping = {'physical': 1, 'technical': 2, 'tactical': 3, 'attacking': 4}
        
        # Initialize new columns
        df['league_pair'] = ''
        df['style_adaptation_required'] = 0
        
        # For players who will transfer, create interaction features with target league
        if 'target_league' in df.columns and 'will_transfer' in df.columns:
            transfer_players = df[df['will_transfer'] == 1].copy()
            
            if len(transfer_players) > 0:
                # Current league vs target league style differences
                for idx, row in transfer_players.iterrows():
                    current_league = row['league']
                    target_league = row['target_league']
                    
                    if target_league:
                        # Create league-pair interaction features
                        df.at[idx, 'league_pair'] = f"{current_league}_to_{target_league}"
                        
                        # Style adaptation required
                        current_style = df.at[idx, 'league_style']
                        target_style = league_characteristics.get(target_league, {}).get('style', 'medium')
                        target_style_num = style_mapping.get(target_style, 0)
                        
                        df.at[idx, 'style_adaptation_required'] = abs(current_style - target_style_num)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate engineered features."""
        
        print("Cleaning engineered features...")
        
        # Handle infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with appropriate defaults
        for col in numeric_cols:
            if col.endswith('_per90') or col.endswith('_rate') or col.endswith('_efficiency'):
                df[col] = df[col].fillna(0)
            elif col.endswith('_pct') or col.endswith('_percentile'):
                df[col] = df[col].fillna(0.5)  # Median percentile
            elif 'share' in col:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def prepare_features_for_modeling(self, df: pd.DataFrame, 
                                    target_column: str = 'will_transfer') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning models.
        
        Args:
            df: Engineered dataset
            target_column: Name of target variable
            
        Returns:
            Tuple of (features_df, target_series)
        """
        print("Preparing features for modeling...")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()
        
        # Remove non-feature columns
        non_feature_cols = [
            'player', 'player_id', 'team', 'season', 'transfer_season',
            'target_league', 'transfer_type'
        ]
        
        cols_to_remove = [col for col in non_feature_cols if col in X.columns]
        X = X.drop(columns=cols_to_remove)
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            if 'numerical' not in self.scalers:
                self.scalers['numerical'] = StandardScaler()
                X[numerical_cols] = self.scalers['numerical'].fit_transform(X[numerical_cols])
            else:
                X[numerical_cols] = self.scalers['numerical'].transform(X[numerical_cols])
        
        print(f"Prepared {len(X.columns)} features for modeling")
        return X, y

if __name__ == "__main__":
    engineer = FeatureEngineer()
    print("Feature engineer ready")
