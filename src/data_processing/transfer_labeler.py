"""
Transfer Labeling System
Identifies and labels player transfers between leagues and clubs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TransferLabeler:
    """Identify and label player transfers for machine learning."""
    
    def __init__(self, min_games_threshold: int = 3, min_minutes_threshold: int = 270):
        """
        Initialize transfer labeler.
        
        Args:
            min_games_threshold: Minimum games to consider a "real" appearance
            min_minutes_threshold: Minimum minutes to consider meaningful playing time
        """
        self.min_games = min_games_threshold
        self.min_minutes = min_minutes_threshold
        
        # Season order for chronological analysis
        self.season_order = [
            '2017-2018', '2018-2019', '2019-2020', '2020-2021',
            '2021-2022', '2022-2023', '2023-2024'
        ]
        
        print(f"TransferLabeler initialized (min_games={min_games_threshold}, min_minutes={min_minutes_threshold})")
    
    def identify_player_movements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify all player movements between clubs and leagues.
        
        Args:
            df: Combined European dataset
            
        Returns:
            DataFrame with player movement records
        """
        print("Identifying player movements across seasons...")
        
        # Sort by player and season for chronological order
        df_sorted = df.sort_values(['player', 'season']).copy()
        
        # Filter to players with meaningful playing time
        significant_players = df_sorted[
            (df_sorted['games'].fillna(0) >= self.min_games) |
            (df_sorted['minutes'].fillna(0) >= self.min_minutes)
        ].copy()
        
        movements = []
        
        for player_name, player_group in significant_players.groupby('player'):
            player_seasons = player_group.sort_values('season')
            
            # Track movements for this player
            previous_row = None
            
            for idx, current_row in player_seasons.iterrows():
                if previous_row is not None:
                    # Check if player moved
                    club_change = previous_row['team'] != current_row['team']
                    league_change = previous_row['league'] != current_row['league']
                    
                    if club_change or league_change:
                        movement = {
                            'player': player_name,
                            'player_id': current_row.get('player_id', ''),
                            'from_season': previous_row['season'],
                            'to_season': current_row['season'],
                            'from_club': previous_row['team'],
                            'to_club': current_row['team'],
                            'from_league': previous_row['league'],
                            'to_league': current_row['league'],
                            'club_change': club_change,
                            'league_change': league_change,
                            'age_at_move': current_row.get('age', np.nan),
                            'from_games': previous_row.get('games', 0),
                            'from_minutes': previous_row.get('minutes', 0),
                            'from_goals': previous_row.get('goals', 0),
                            'from_assists': previous_row.get('assists', 0),
                            'to_games': current_row.get('games', 0),
                            'to_minutes': current_row.get('minutes', 0),
                            'to_goals': current_row.get('goals', 0),
                            'to_assists': current_row.get('assists', 0),
                            'nationality': current_row.get('nationality', ''),
                            'position': current_row.get('position', '')
                        }
                        movements.append(movement)
                
                previous_row = current_row
        
        movements_df = pd.DataFrame(movements)
        
        if not movements_df.empty:
            # Add derived features
            movements_df['performance_change'] = (
                (movements_df['to_goals'] + movements_df['to_assists']) - 
                (movements_df['from_goals'] + movements_df['from_assists'])
            )
            
            movements_df['minutes_change'] = (
                movements_df['to_minutes'] - movements_df['from_minutes']
            )
            
            # Categorize movement types
            movements_df['movement_type'] = movements_df.apply(self._categorize_movement, axis=1)
        
        print(f"Identified {len(movements_df)} player movements")
        return movements_df
    
    def _categorize_movement(self, row) -> str:
        """Categorize the type of movement."""
        if row['league_change'] and row['club_change']:
            return 'cross_league_transfer'
        elif row['league_change'] and not row['club_change']:
            return 'league_promotion_relegation'
        elif not row['league_change'] and row['club_change']:
            return 'domestic_transfer'
        else:
            return 'unknown'
    
    def create_transfer_labels(self, df: pd.DataFrame, look_ahead_seasons: int = 1) -> pd.DataFrame:
        """
        Create transfer labels for machine learning.
        
        Args:
            df: Combined European dataset
            look_ahead_seasons: How many seasons ahead to predict
            
        Returns:
            DataFrame with transfer labels added
        """
        print(f"Creating transfer labels (look ahead: {look_ahead_seasons} seasons)")
        
        # First get all movements
        movements_df = self.identify_player_movements(df)
        
        # Create a copy for labeling
        labeled_df = df.copy()
        
        # Initialize label columns
        labeled_df['will_transfer'] = 0  # Binary: will transfer in next season(s)
        labeled_df['transfer_type'] = 'stay'  # Type of transfer
        labeled_df['target_league'] = ''  # Target league if transferring
        labeled_df['transfer_season'] = ''  # When the transfer happens
        
        # For each player-season, check if they transfer in the future
        for idx, row in labeled_df.iterrows():
            player = row['player']
            current_season = row['season']
            
            # Find future movements for this player
            future_movements = movements_df[
                (movements_df['player'] == player) &
                (movements_df['from_season'] == current_season)
            ]
            
            if not future_movements.empty:
                # Player transfers after this season
                transfer = future_movements.iloc[0]
                
                labeled_df.at[idx, 'will_transfer'] = 1
                labeled_df.at[idx, 'transfer_type'] = transfer['movement_type']
                labeled_df.at[idx, 'target_league'] = transfer['to_league']
                labeled_df.at[idx, 'transfer_season'] = transfer['to_season']
        
        print(f"Added transfer labels to {len(labeled_df)} records")
        transfer_rate = labeled_df['will_transfer'].mean()
        print(f"Transfer rate: {transfer_rate:.1%}")
        
        return labeled_df

if __name__ == "__main__":
    labeler = TransferLabeler()
    print("Transfer labeler ready")
