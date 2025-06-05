"""
Transfer Prediction Models
Machine learning models for predicting player transfers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

class TransferPredictor:
    """Machine learning pipeline for transfer prediction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize transfer predictor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.model_scores = {}
        
        print(f"TransferPredictor initialized (random_state={random_state})")
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, stratify: bool = True) -> Tuple:
        """
        Prepare data for training and testing.
        
        Args:
            X: Features dataframe
            y: Target series
            test_size: Fraction of data to use for testing
            stratify: Whether to stratify the split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"Preparing data (test_size={test_size}, stratify={stratify})")
        
        # Check class balance
        class_counts = y.value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        # Handle class imbalance if severe
        minority_class_ratio = class_counts.min() / class_counts.sum()
        if minority_class_ratio < 0.05:
            print(f"Severe class imbalance detected (minority class: {minority_class_ratio:.1%})")
        
        # Split data
        stratify_param = y if stratify and len(y.unique()) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def build_baseline_models(self) -> Dict:
        """Build baseline machine learning models."""
        
        print("Building baseline models...")
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                learning_rate=0.1
            )
        }
        
        self.models.update(models)
        print(f"Built {len(models)} baseline models")
        return models
    
    def build_ensemble_models(self) -> Dict:
        """Build ensemble models combining multiple base models."""
        
        print("Building ensemble models...")
        
        # Voting classifier with multiple models
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced')),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
                ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced'))
            ],
            voting='soft'
        )
        
        # Calibrated classifier for better probability estimates
        calibrated_rf = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced'),
            cv=3
        )
        
        ensemble_models = {
            'voting_classifier': voting_clf,
            'calibrated_rf': calibrated_rf
        }
        
        self.models.update(ensemble_models)
        print(f"Built {len(ensemble_models)} ensemble models")
        return ensemble_models
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train all models and return training scores."""
        
        print("Training all models...")
        
        trained_models = {}
        training_scores = {}
        
        for name, model in self.models.items():
            try:
                print(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                training_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }
                
                print(f"{name} CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        self.models = trained_models
        self.model_scores = training_scores
        
        print(f"Successfully trained {len(trained_models)} models")
        return training_scores
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all trained models on test data."""
        
        print("Evaluating models on test data...")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                results = {
                    'accuracy': (y_pred == y_test).mean(),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                evaluation_results[name] = results
                
                # Log key metrics
                auc_str = f"{results['roc_auc']:.3f}" if results['roc_auc'] is not None else "N/A"
                print(f"{name} - Accuracy: {results['accuracy']:.3f}, AUC: {auc_str}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                continue
        
        return evaluation_results
    
    def select_best_model(self, evaluation_results: Dict, metric: str = 'roc_auc') -> str:
        """Select the best performing model based on specified metric."""
        
        print(f"Selecting best model based on {metric}...")
        
        best_score = -1
        best_model_name = None
        
        for name, results in evaluation_results.items():
            score = results.get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_model_name = name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]
            print(f"Best model: {best_model_name} ({metric}={best_score:.3f})")
        else:
            print("Could not select best model")
        
        return best_model_name
    
    def analyze_feature_importance(self, X: pd.DataFrame, model_name: Optional[str] = None) -> Dict:
        """Analyze feature importance for tree-based models."""
        
        if model_name is None:
            model_name = 'random_forest'  # Default to random forest
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return {}
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature_importances_ attribute")
            return {}
        
        # Get feature importance
        importance_scores = model.feature_importances_
        feature_names = X.columns.tolist()
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        self.feature_importance[model_name] = {
            'importance_scores': importance_dict,
            'top_features': sorted_importance[:20],  # Top 20 features
            'feature_names': feature_names
        }
        
        print(f"Analyzed feature importance for {model_name}")
        return self.feature_importance[model_name]
    
    def predict_transfers(self, X: pd.DataFrame, return_probabilities: bool = True) -> Dict:
        """Make transfer predictions using the best model."""
        
        if self.best_model is None:
            raise ValueError("No best model selected. Run model evaluation first.")
        
        predictions = self.best_model.predict(X)
        
        results = {
            'predictions': predictions,
            'transfer_probability': None
        }
        
        if return_probabilities and hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X)
            results['transfer_probability'] = probabilities[:, 1]  # Probability of transfer
        
        print(f"Made predictions for {len(X)} players")
        return results
    
    def save_models(self, output_dir: str = "models/trained"):
        """Save trained models to disk."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_file = output_path / f"{name}_model.pkl"
            joblib.dump(model, model_file)
            print(f"Saved {name} to {model_file}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_file = output_path / "best_transfer_model.pkl"
            joblib.dump(self.best_model, best_model_file)
            print(f"Saved best model to {best_model_file}")
        
        # Save metadata
        metadata = {
            'models_trained': list(self.models.keys()),
            'training_scores': self.model_scores,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = output_path / "model_metadata.pkl"
        joblib.dump(metadata, metadata_file)
        print(f"Saved model metadata to {metadata_file}")
        
        return output_path
    
    def generate_model_report(self, evaluation_results: Dict, output_file: str = None) -> str:
        """Generate comprehensive model performance report."""
        
        report_lines = [
            "TRANSFER PREDICTION MODEL REPORT",
            "=" * 50,
            "",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Models Evaluated: {len(evaluation_results)}",
            ""
        ]
        
        # Model performance summary
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 30)
        
        for name, results in evaluation_results.items():
            accuracy = results.get('accuracy', 0)
            auc = results.get('roc_auc', 0)
            report_lines.append(f"{name}:")
            report_lines.append(f"  Accuracy: {accuracy:.3f}")
            auc_str = f"{auc:.3f}" if auc else "N/A"
            report_lines.append(f"  ROC AUC:  {auc_str}")
            report_lines.append("")
        
        # Feature importance (if available)
        if self.feature_importance:
            report_lines.append("TOP FEATURES (Random Forest)")
            report_lines.append("-" * 30)
            
            rf_importance = self.feature_importance.get('random_forest', {})
            top_features = rf_importance.get('top_features', [])
            
            for i, (feature, importance) in enumerate(top_features[:10], 1):
                report_lines.append(f"{i:2d}. {feature}: {importance:.4f}")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_text)
            
            print(f"Saved model report to {output_path}")
        
        return report_text

if __name__ == "__main__":
    predictor = TransferPredictor()
    print("Transfer predictor ready")
