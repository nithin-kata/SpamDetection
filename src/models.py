"""
Machine learning models for text classification.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

class TextClassifier:
    """Wrapper class for text classification models."""
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize classifier with specified model type.
        
        Args:
            model_type (str): Type of model ('naive_bayes', 'logistic_regression', 
                             'random_forest', 'svm')
        """
        self.model_type = model_type
        self.model = self._get_model(model_type)
        self.is_fitted = False
    
    def _get_model(self, model_type):
        """Get the specified model instance."""
        models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True)
        }
        
        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return models[model_type]
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance (for applicable models)."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = abs(self.model.coef_[0])
        else:
            return None
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        return importances
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)
        self.is_fitted = True

class ModelTuner:
    """Handles hyperparameter tuning for models."""
    
    def __init__(self):
        self.param_grids = {
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def tune_model(self, model_type, X_train, y_train, cv=5, scoring='f1_macro'):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_type (str): Type of model to tune
            X_train: Training features
            y_train: Training labels
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
        
        Returns:
            Best model and parameters
        """
        classifier = TextClassifier(model_type)
        param_grid = self.param_grids.get(model_type, {})
        
        if not param_grid:
            print(f"No parameter grid defined for {model_type}")
            return classifier.train(X_train, y_train)
        
        grid_search = GridSearchCV(
            classifier.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update classifier with best model
        classifier.model = grid_search.best_estimator_
        classifier.is_fitted = True
        
        print(f"Best parameters for {model_type}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return classifier, grid_search.best_params_