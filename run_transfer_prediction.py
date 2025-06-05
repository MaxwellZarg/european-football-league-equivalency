#!/usr/bin/env python3
"""
Transfer Prediction Execution Script
Main script to run the European transfer prediction system
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

# Import pipeline
from main_pipeline import TransferPredictionPipeline

def setup_logging(log_level='INFO'):
    """Setup logging configuration."""
    
    # Create logs directory
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_file = log_dir / f"transfer_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

def run_pipeline(args):
    """Run the main transfer prediction pipeline."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting transfer prediction pipeline...")
    
    # Initialize pipeline
    pipeline = TransferPredictionPipeline(
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    
    # Parse leagues argument
    leagues = None
    if args.leagues:
        leagues = [league.strip() for league in args.leagues.split(',')]
        logger.info(f"Selected leagues: {leagues}")
    
    try:
        # Run complete pipeline
        logger.info("Executing complete pipeline...")
        results = pipeline.run_complete_pipeline(
            leagues=leagues,
            save_intermediate=True
        )
        
        # Print summary
        print("\n" + "="*60)
        print("TRANSFER PREDICTION PIPELINE COMPLETED!")
        print("="*60)
        
        data_summary = results['data_summary']
        print(f"Total Players Analyzed: {data_summary['total_players']:,}")
        print(f"Total Features Created: {data_summary['total_features']}")
        print(f"Transfer Rate: {data_summary['transfer_rate']:.1%}")
        print(f"Leagues Included: {', '.join(data_summary['leagues_included'])}")
        
        if 'best_model' in results and results['best_model']:
            best_model = results['best_model']
            model_results = results['model_performance'][best_model]
            print(f"\nBest Model: {best_model}")
            print(f"Model Accuracy: {model_results['accuracy']:.1%}")
            print(f"ROC AUC Score: {model_results['roc_auc']:.3f}")
        
        print(f"\nResults saved to: {pipeline.output_dir}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nPipeline failed: {str(e)}")
        return None

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="European Football Transfer Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with all leagues
  python run_transfer_prediction.py
  
  # Run with specific leagues only
  python run_transfer_prediction.py --leagues "premier_league,la_liga,serie_a"
  
  # Test mode with sample data
  python run_transfer_prediction.py --test-mode
        """
    )
    
    parser.add_argument(
        '--data-root',
        default='data/raw',
        help='Path to raw data directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--leagues',
        help='Comma-separated list of leagues to include (default: all)'
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
    
    print("EUROPEAN FOOTBALL TRANSFER PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Directory: {args.output_dir}")
    
    if args.test_mode:
        print("Running in TEST MODE")
        args.leagues = "premier_league"  # Use only Premier League for testing
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    print("\nStarting transfer prediction pipeline...")
    results = run_pipeline(args)
    
    if results is None:
        print("\nPipeline execution failed!")
        return 1
    
    print(f"\nPipeline completed successfully!")
    print(f"Check {args.output_dir} for detailed results and reports.")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
