"""
Main Transfer Prediction Pipeline
Orchestrates the complete transfer prediction workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Import our custom modules
from data_processing.european_data_loader import EuropeanDataLoader
from data_processing.transfer_labeler import TransferLabeler
from data_processing.feature_engineer import FeatureEngineer
from transfer_prediction.transfer_models import TransferPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/transfer_prediction_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class TransferPredictionPipeline:
    """Complete pipeline for European transfer prediction."""
    
    def __init__(self, data_root: str = "data/raw", output_dir: str = "results"):
        """
        Initialize the transfer prediction pipeline.
        
        Args:
            data_root: Path to raw data directory
            output_dir: Path to save results
        """
        self.data_root = data_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = EuropeanDataLoader()
        self.transfer_labeler = TransferLabeler()
        self.feature_engineer = FeatureEngineer()
        self.predictor = TransferPredictor()
        
        # Pipeline state
        self.european_data = None
        self.labeled_data = None
        self.engineered_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        print(f"TransferPredictionPipeline initialized")
        print(f"Data root: {self.data_root}")
        print(f"Output directory: {self.output_dir}")
    
    def run_complete_pipeline(self, leagues: list = None, save_intermediate: bool = True) -> dict:
        """
        Run the complete transfer prediction pipeline.
        
        Args:
            leagues: List of leagues to include (None for all)
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with pipeline results and metrics
        """
        print("Starting complete transfer prediction pipeline...")
        
        try:
            # Step 1: Load European data
            print("Step 1: Loading European football data...")
            self.european_data = self.data_loader.load_all_european_data(
                data_root=self.data_root,
                leagues=leagues
            )
            
            if save_intermediate:
                data_file = self.data_loader.save_combined_dataset(
                    self.european_data,
                    self.output_dir / "processed" / "european_combined.csv"
                )
            
            # Step 2: Label transfers
            print("Step 2: Identifying and labeling transfers...")
            self.labeled_data = self.transfer_labeler.create_transfer_labels(self.european_data)
            
            if save_intermediate:
                labeled_file = self.output_dir / "processed" / "transfer_labeled.csv"
                labeled_file.parent.mkdir(parents=True, exist_ok=True)
                self.labeled_data.to_csv(labeled_file, index=False)
                print(f"Saved labeled data to {labeled_file}")
            
            # Step 3: Engineer features
            print("Step 3: Engineering features for machine learning...")
            self.engineered_data = self.feature_engineer.engineer_all_features(self.labeled_data)
            
            if save_intermediate:
                features_file = self.output_dir / "processed" / "engineered_features.csv"
                features_file.parent.mkdir(parents=True, exist_ok=True)
                self.engineered_data.to_csv(features_file, index=False)
                print(f"Saved engineered features to {features_file}")
            
            # Step 4: Prepare data for modeling
            print("Step 4: Preparing data for machine learning...")
            X, y = self.feature_engineer.prepare_features_for_modeling(
                self.engineered_data,
                target_column='will_transfer'
            )
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = self.predictor.prepare_data(X, y)
            
            # Step 5: Build and train models
            print("Step 5: Building and training machine learning models...")
            
            # Build all model types
            self.predictor.build_baseline_models()
            self.predictor.build_ensemble_models()
            
            # Train all models
            training_scores = self.predictor.train_all_models(self.X_train, self.y_train)
            
            # Step 6: Evaluate models
            print("Step 6: Evaluating model performance...")
            evaluation_results = self.predictor.evaluate_models(self.X_test, self.y_test)
            
            # Select best model
            best_model_name = self.predictor.select_best_model(evaluation_results)
            
            # Step 7: Analyze results
            print("Step 7: Analyzing feature importance and model insights...")
            feature_importance = self.predictor.analyze_feature_importance(X)
            
            # Step 8: Save models and results
            print("Step 8: Saving models and generating reports...")
            model_dir = self.predictor.save_models(self.output_dir / "models")
            
            # Generate comprehensive report
            report = self.predictor.generate_model_report(
                evaluation_results,
                self.output_dir / "reports" / "model_performance_report.txt"
            )
            
            # Pipeline summary
            pipeline_results = {
                'data_summary': {
                    'total_players': len(self.european_data),
                    'total_features': len(X.columns),
                    'transfer_rate': self.y_train.mean(),
                    'leagues_included': self.european_data['league'].unique().tolist()
                },
                'model_performance': evaluation_results,
                'best_model': best_model_name,
                'feature_importance': feature_importance,
                'training_scores': training_scores
            }
            
            # Save pipeline summary
            self._save_pipeline_summary(pipeline_results)
            
            print("Transfer prediction pipeline completed successfully!")
            return pipeline_results
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise e
    
    def predict_new_transfers(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict transfers for new player data.
        
        Args:
            new_data: DataFrame with player statistics
            
        Returns:
            DataFrame with transfer predictions
        """
        print(f"Making transfer predictions for {len(new_data)} players...")
        
        # Engineer features for new data
        engineered_new = self.feature_engineer.engineer_all_features(new_data)
        
        # Prepare features
        X_new, _ = self.feature_engineer.prepare_features_for_modeling(
            engineered_new,
            target_column='will_transfer' if 'will_transfer' in engineered_new.columns else None
        )
        
        # Make predictions
        predictions = self.predictor.predict_transfers(X_new, return_probabilities=True)
        
        # Combine with original data
        results_df = new_data.copy()
        results_df['transfer_prediction'] = predictions['predictions']
        
        if predictions['transfer_probability'] is not None:
            results_df['transfer_probability'] = predictions['transfer_probability']
            
            # Add risk categories
            results_df['transfer_risk'] = pd.cut(
                results_df['transfer_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
        
        print("Transfer predictions completed")
        return results_df
    
    def get_transfer_summary(self) -> dict:
        """Get summary statistics of transfer patterns."""
        
        if self.labeled_data is None:
            raise ValueError("No labeled data available. Run pipeline first.")
        
        transfer_data = self.labeled_data[self.labeled_data['will_transfer'] == 1]
        
        summary = {
            'total_players': self.labeled_data['player'].nunique(),
            'total_transfers': len(transfer_data),
            'transfer_rate': len(transfer_data) / len(self.labeled_data) if len(self.labeled_data) > 0 else 0,
            'transfers_by_league': transfer_data['league'].value_counts().to_dict(),
            'transfers_by_type': transfer_data['transfer_type'].value_counts().to_dict(),
            'target_leagues': transfer_data['target_league'].value_counts().to_dict(),
            'avg_age_at_transfer': transfer_data['age'].mean() if 'age' in transfer_data.columns else None,
            'transfers_by_season': transfer_data['season'].value_counts().to_dict()
        }
        
        return summary
    
    def _save_pipeline_summary(self, results: dict):
        """Save pipeline execution summary."""
        
        summary_file = self.output_dir / "pipeline_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("EUROPEAN TRANSFER PREDICTION PIPELINE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Pipeline executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            data_summary = results['data_summary']
            f.write("DATA SUMMARY:\n")
            f.write(f"  Total Players: {data_summary['total_players']:,}\n")
            f.write(f"  Total Features: {data_summary['total_features']}\n")
            f.write(f"  Transfer Rate: {data_summary['transfer_rate']:.1%}\n")
            f.write(f"  Leagues: {', '.join(data_summary['leagues_included'])}\n\n")
            
            # Model performance
            f.write("MODEL PERFORMANCE:\n")
            best_model = results['best_model']
            if best_model and best_model in results['model_performance']:
                best_results = results['model_performance'][best_model]
                f.write(f"  Best Model: {best_model}\n")
                f.write(f"  Accuracy: {best_results['accuracy']:.3f}\n")
                f.write(f"  ROC AUC: {best_results['roc_auc']:.3f}\n\n")
            
            # Top features
            if 'feature_importance' in results and results['feature_importance']:
                f.write("TOP 10 FEATURES:\n")
                top_features = results['feature_importance'].get('top_features', [])
                for i, (feature, importance) in enumerate(top_features[:10], 1):
                    f.write(f"  {i:2d}. {feature}: {importance:.4f}\n")
        
        print(f"Pipeline summary saved to {summary_file}")

def main():
    """Run the complete transfer prediction pipeline."""
    
    print("EUROPEAN TRANSFER PREDICTION PIPELINE")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = TransferPredictionPipeline()
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(save_intermediate=True)
        
        # Get transfer summary
        transfer_summary = pipeline.get_transfer_summary()
        
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Total Players Analyzed: {results['data_summary']['total_players']:,}")
        print(f"Best Model: {results['best_model']}")
        print(f"Model Accuracy: {results['model_performance'][results['best_model']]['accuracy']:.1%}")
        print(f"Transfer Rate: {results['data_summary']['transfer_rate']:.1%}")
        print(f"\nResults saved to: {pipeline.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        logger.error(f"Pipeline execution failed: {str(e)}")
        return None

if __name__ == "__main__":
    results = main()
