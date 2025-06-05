"""
Transfer Labeling System
Identifies and labels player transfers between leagues and clubs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
        
        logger.info(f"TransferLabeler initialized (min_games={min_games_threshold}, min_minutes={min_minutes_threshold})")
    
    def identify_player_movements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify all player movements between clubs and leagues.
        
        Args:
            df: Combined European dataset
            
        Returns:
            DataFrame with player movement records
        """
        logger.info("Identifying player movements across seasons...")
        
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
        
        logger.info(f"Identified {len(movements_df)} player movements")
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
        logger.info(f"Creating transfer labels (look ahead: {look_ahead_seasons} seasons)")
        
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
            current_league = row['league']
            current_club = row['team']
            
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
        
        logger.info(f"Added transfer labels to {len(labeled_df)} records")
        return labeled_df
    
    def create_transfer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that might predict transfers.
        
        Args:
            df: Dataset with transfer labels
            
        Returns:
            DataFrame with additional transfer prediction features
        """
        logger.info("Creating transfer prediction features...")
        
        featured_df = df.copy()
        
        # Performance features
        self._add_performance_features(featured_df)
        
        # Career trajectory features
        self._add_trajectory_features(featured_df)
        
        # Contract and market features
        self._add_market_features(featured_df)
        
        # Team context features
        self._add_team_context_features(featured_df)
        
        logger.info(f"Added transfer prediction features")
        return featured_df
    
    def _add_performance_features(self, df: pd.DataFrame) -> None:
        """Add performance-based features."""
        
        # Goals + Assists per 90
        if 'minutes_90s' in df.columns:
            df['goal_contributions_per90'] = (
                (df['goals'].fillna(0) + df['assists'].fillna(0)) / 
                df['minutes_90s'].replace(0, np.nan)
            )
        
        # Performance percentiles within league-season
        for metric in ['goals', 'assists', 'minutes', 'games']:
            if metric in df.columns:
                df[f'{metric}_league_percentile'] = df.groupby(['league', 'season'])[metric].rank(pct=True)
        
        # Expected vs actual performance
        if 'xg' in df.columns and 'goals' in df.columns:
            df['goals_vs_xg'] = df['goals'] - df['xg']
        
        if 'xg_assist' in df.columns and 'assists' in df.columns:
            df['assists_vs_xa'] = df['assists'] - df['xg_assist']
    
    def _add_trajectory_features(self, df: pd.DataFrame) -> None:
        """Add career trajectory features."""
        
        # Age-based features
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['is_peak_age'] = ((df['age'] >= 25) & (df['age'] <= 29)).astype(int)
            df['is_young_talent'] = (df['age'] <= 23).astype(int)
            df['is_veteran'] = (df['age'] >= 32).astype(int)
        
        # Playing time trend (simplified - would need multi-season data for proper implementation)
        if 'minutes' in df.columns:
            df['minutes_log'] = np.log1p(df['minutes'])
            df['is_regular_starter'] = (df['minutes'] >= 2000).astype(int)
            df['is_fringe_player'] = (df['minutes'] <= 500).astype(int)
    
    def _add_market_features(self, df: pd.DataFrame) -> None:
        """Add market-related features."""
        
        # League prestige ranking (could be enhanced with UEFA coefficients)
        league_prestige = {
            'premier_league': 1.0,
            'la_liga': 0.95,
            'bundesliga': 0.90,
            'serie_a': 0.85,
            'ligue_1': 0.80,
            'primeira_liga': 0.70,
            'eredivisie': 0.65
        }
        
        df['league_prestige'] = df['league'].map(league_prestige).fillna(0.5)
        
        # Contract situation proxies
        df['likely_contract_year'] = 0  # Placeholder - would need contract data
        
        # Performance vs league average
        for metric in ['goals', 'assists']:
            if metric in df.columns:
                league_avg = df.groupby(['league', 'season'])[metric].transform('mean')
                df[f'{metric}_vs_league_avg'] = df[metric] - league_avg
    
    def _add_team_context_features(self, df: pd.DataFrame) -> None:
        """Add team context features."""
        
        # Team performance context
        team_stats = df.groupby(['team', 'season']).agg({
            'goals': 'sum',
            'assists': 'sum',
            'minutes': 'sum',
            'player': 'count'
        }).reset_index()
        
        team_stats.columns = ['team', 'season', 'team_total_goals', 'team_total_assists', 
                             'team_total_minutes', 'team_squad_size']
        
        df = df.merge(team_stats, on=['team', 'season'], how='left')
        
        # Player importance to team
        if 'team_total_goals' in df.columns:
            df['goal_share'] = df['goals'] / df['team_total_goals'].replace(0, np.nan)
            df['assist_share'] = df['assists'] / df['team_total_assists'].replace(0, np.nan)
            df['minutes_share'] = df['minutes'] / df['team_total_minutes'].replace(0, np.nan)
    
    def get_transfer_summary(self, labeled_df: pd.DataFrame) -> Dict:
        """Get summary statistics of transfer patterns."""
        
        transfer_stats = labeled_df[labeled_df['will_transfer'] == 1]
        
        summary = {
            'total_players': labeled_df['player'].nunique(),
            'total_transfers': len(transfer_stats),
            'transfer_rate': len(transfer_stats) / len(labeled_df) if len(labeled_df) > 0 else 0,
            'transfers_by_league': transfer_stats['league'].value_counts().to_dict(),
            'transfers_by_type': transfer_stats['transfer_type'].value_counts().to_dict(),
            'target_leagues': transfer_stats['target_league'].value_counts().to_dict(),
            'avg_age_at_transfer': transfer_stats['age'].mean() if 'age' in transfer_stats.columns else None,
            'transfers_by_season': transfer_stats['season'].value_counts().to_dict()
        }
        
        return summary
    
    def save_labeled_dataset(self, labeled_df: pd.DataFrame, 
                           output_path: str = "data/processed/transfer_labeled_dataset.csv"):
        """Save the labeled dataset for machine learning."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        labeled_df.to_csv(output_file, index=False)
        logger.info(f"Saved labeled dataset to {output_file}")
        
        # Save summary statistics
        summary = self.get_transfer_summary(labeled_df)
        summary_file = output_file.parent / "transfer_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("TRANSFER DATASET SUMMARY\n")
            f.write("========================\n\n")
            f.write(f"Total Players: {summary['total_players']:,}\n")
            f.write(f"Total Transfers: {summary['total_transfers']:,}\n")
            f.write(f"Transfer Rate: {summary['transfer_rate']:.1%}\n\n")
            
            f.write("Transfers by League:\n")
            for league, count in summary['transfers_by_league'].items():
                f.write(f"  {league}: {count:,}\n")
            
            f.write("\nTransfers by Type:\n")
            for transfer_type, count in summary['transfers_by_type'].items():
                f.write(f"  {transfer_type}: {count:,}\n")
        
        logger.info(f"Saved transfer summary to {summary_file}")
        return output_file


def main():
    """Test the Transfer Labeler"""
    
    print("Testing Transfer Labeler")
    print("=" * 40)
    
    # This would be called with real data
    # labeler = TransferLabeler()
    # labeled_data = labeler.create_transfer_labels(european_data)
    
    print("Transfer Labeler ready for use")


if __name__ == "__main__":
    main()
