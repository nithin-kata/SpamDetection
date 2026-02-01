"""
Model evaluation utilities and metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True test labels
            model_name (str): Name for the model
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_score_macro': f1_score(y_test, y_pred, average='macro')
        }
        
        # Add AUC score for binary classification
        if len(np.unique(y_test)) == 2:
            metrics['auc_score'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def print_evaluation_report(self, model_name):
        """Print detailed evaluation report."""
        if model_name not in self.results:
            print(f"No results found for model: {model_name}")
            return
        
        result = self.results[model_name]
        metrics = result['metrics']
        
        print(f"\n{'='*50}")
        print(f"EVALUATION REPORT: {model_name}")
        print(f"{'='*50}")
        
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (weighted): {metrics['precision']:.4f}")
        print(f"Recall (weighted):    {metrics['recall']:.4f}")
        print(f"F1-Score (weighted):  {metrics['f1_score']:.4f}")
        print(f"Precision (macro):    {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):       {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):     {metrics['f1_score_macro']:.4f}")
        
        if 'auc_score' in metrics:
            print(f"AUC Score:           {metrics['auc_score']:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(result['y_true'], result['y_pred']))
    
    def plot_confusion_matrix(self, model_name, class_names=None, figsize=(8, 6)):
        """Plot confusion matrix."""
        if model_name not in self.results:
            print(f"No results found for model: {model_name}")
            return
        
        result = self.results[model_name]
        cm = confusion_matrix(result['y_true'], result['y_pred'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, model_name, figsize=(8, 6)):
        """Plot ROC curve for binary classification."""
        if model_name not in self.results:
            print(f"No results found for model: {model_name}")
            return
        
        result = self.results[model_name]
        
        # Check if binary classification
        if len(np.unique(result['y_true'])) != 2:
            print("ROC curve is only available for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'][:, 1])
        auc_score = roc_auc_score(result['y_true'], result['y_pred_proba'][:, 1])
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """Compare multiple models and return comparison DataFrame."""
        if not self.results:
            print("No model results to compare")
            return None
        
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics'].copy()
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        df = df.set_index('model_name')
        
        return df
    
    def plot_model_comparison(self, metrics=['accuracy', 'precision', 'recall', 'f1_score'], 
                            figsize=(12, 8)):
        """Plot comparison of multiple models."""
        df = self.compare_models()
        if df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics[:4]):
            if metric in df.columns:
                df[metric].plot(kind='bar', ax=axes[i], title=f'{metric.title()}')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, metric='f1_score'):
        """Get the best performing model based on specified metric."""
        if not self.results:
            print("No model results available")
            return None
        
        best_score = -1
        best_model = None
        
        for model_name, result in self.results.items():
            score = result['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score