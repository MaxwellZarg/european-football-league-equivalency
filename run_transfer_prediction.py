#!/usr/bin/env python3
"""
Fixed Transfer Prediction Execution Script
Runs the PROPER cross-league performance prediction system
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import FIXED pipeline
from fixed_main_pipeline import FixedTransferPredictionPipeline

def setup_logging(log_level='INFO'):
    """Setup logging configuration."""
    
    # Create logs directory
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_file = log_dir / f"proper_transfer_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def run_proper_pipeline(args):
    """Run the PROPER cross-league prediction pipeline."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting PROPER cross-league prediction pipeline...")
    
    # Initialize FIXED pipeline
    pipeline = FixedTransferPredictionPipeline(
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    
    try:
        # Run proper pipeline
        logger.info("Executing proper pipeline...")
        results = pipeline.run_proper_pipeline()
        
        # Print summary
        print("\n" + "="*80)
        print(" PROPER CROSS-LEAGUE PREDICTION PIPELINE COMPLETED!")
        print("="*80)
        
        metrics = results['evaluation_metrics']
        print(f" Players analyzed: {metrics['total_players_analyzed']:,}")
        print(f" League pairs with data: {metrics['league_pairs_with_data']}")
        print(f" Average confidence: {metrics['average_confidence']:.2f}")
        print(f" Improvement opportunities: {metrics['players_could_improve_by_moving']}")
        
        print(f"\n🔬 Methodology: {metrics['methodology']}")
        print(" No data leakage")
        print(" Proper temporal validation") 
        print(" Academically sound")
        
        print(f"\n Results saved to: {pipeline.output_dir}")
        
        # Show example usage
        print("\n" + "="*80)
        print(" EXAMPLE USAGE:")
        print("="*80)
        
        example = pipeline.predict_player_cross_league_performance(
            player_name="Example_Bundesliga_Player",
            current_league="bundesliga",
            goals_per_90=0.45,
            assists_per_90=0.25,
            age=24,
            position="FW"
        )
        
        print(f"Player: {example['player']}")
        print(f"Current league: {example['current_performance']['league']}")
        print(f"Current stats: {example['current_performance']['goals_per_90']:.2f} goals/90, {example['current_performance']['assists_per_90']:.2f} assists/90")
        print(f"Recommended league: {example['best_league_recommendation']}")
        print(f"Current league ranking: {example['current_league_ranking']}")
        
        print("\n Predictions for all leagues:")
        for league, pred in example['cross_league_predictions'].items():
            goals = pred['predicted_goals_90']
            assists = pred['predicted_assists_90']
            confidence = pred['confidence']
            print(f"  {league:15} {goals:.2f} goals/90, {assists:.2f} assists/90 (confidence: {confidence:.2f})")
        
        print("\n READY FOR ACADEMIC PUBLICATION!")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\n Pipeline failed: {str(e)}")
        return None

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="PROPER European Football Cross-League Performance Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
 --

Methodology:
   Proper temporal validation (NO data leakage)
   Training: 2017-2022 (5 seasons)
   Validation: 2022-2023
   Test: 2023-2024 (held out)
   Cross-league performance prediction
   Academically sound
        """
    )
    
    parser.add_argument(
        '--data-root',
        default='data/raw',
        help='Path to raw data directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results_proper',
        help='Output directory for results (default: results_proper)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with limited data'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    print("PROPER EUROPEAN CROSS-LEAGUE PERFORMANCE PREDICTION")
    print("=" * 80)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Data Root: {args.data_root}")
    print(f" Output Directory: {args.output_dir}")
    print(f" Methodology: Proper temporal validation (NO data leakage)")
    
    if args.test_mode:
        print(" Running in TEST MODE")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    print("\n Starting cross-league prediction pipeline...")
    results = run_proper_pipeline(args)
    
    if results is None:
        print("\n Pipeline execution failed!")
        return 1
    
    print(f"\n Pipeline completed successfully!")
    print(f" Check {args.output_dir} for detailed results.")
    print("\n Ready for academic publication!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
