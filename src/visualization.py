"""
Visualization utilities for text classification results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TextVisualization:
    """Handles visualization of text data and results."""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_class_distribution(self, labels, title="Class Distribution", figsize=(10, 6)):
        """Plot distribution of classes in the dataset."""
        class_counts = Counter(labels)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        ax1.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
        ax1.set_title('Class Distribution (Bar Plot)')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        
        # Pie chart
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (Pie Chart)')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        return class_counts
    
    def plot_text_length_distribution(self, texts, labels, figsize=(12, 8)):
        """Plot distribution of text lengths by class."""
        df = pd.DataFrame({
            'text': texts,
            'label': labels,
            'length': [len(text.split()) for text in texts]
        })
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Overall distribution
        axes[0, 0].hist(df['length'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Overall Text Length Distribution')
        axes[0, 0].set_xlabel('Number of Words')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot by class
        df.boxplot(column='length', by='label', ax=axes[0, 1])
        axes[0, 1].set_title('Text Length by Class')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Number of Words')
        
        # Violin plot
        sns.violinplot(data=df, x='label', y='length', ax=axes[1, 0])
        axes[1, 0].set_title('Text Length Distribution by Class')
        
        # Statistics table
        stats = df.groupby('label')['length'].agg(['mean', 'median', 'std', 'min', 'max'])
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=stats.round(2).values,
                                rowLabels=stats.index,
                                colLabels=stats.columns,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1, 1].set_title('Length Statistics by Class')
        
        plt.tight_layout()
        plt.show()
    
    def create_wordcloud(self, texts, labels=None, class_name=None, figsize=(15, 8)):
        """Create word clouds for text data."""
        if labels is not None and class_name is not None:
            # Filter texts for specific class
            class_texts = [text for text, label in zip(texts, labels) if label == class_name]
            combined_text = ' '.join(class_texts)
            title = f'Word Cloud - {class_name}'
        else:
            combined_text = ' '.join(texts)
            title = 'Word Cloud - All Classes'
        
        if not combined_text.strip():
            print(f"No text data available for {title}")
            return
        
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_importance, top_n=20, figsize=(12, 8)):
        """Plot feature importance from models."""
        if feature_importance is None:
            print("No feature importance data available")
            return
        
        if isinstance(feature_importance, dict):
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_features[:top_n])
        else:
            # Assume it's an array
            indices = np.argsort(feature_importance)[::-1][:top_n]
            features = [f'Feature_{i}' for i in indices]
            importance = feature_importance[indices]
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self, train_scores, val_scores, train_sizes, figsize=(10, 6)):
        """Plot learning curve to analyze model performance vs training size."""
        plt.figure(figsize=figsize)
        
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', 
                label='Training Score', color='blue')
        plt.fill_between(train_sizes, 
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', 
                label='Validation Score', color='red')
        plt.fill_between(train_sizes, 
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def create_interactive_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Create interactive confusion matrix using Plotly."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=class_names, y=class_names,
                       color_continuous_scale='Blues',
                       text_auto=True)
        
        fig.update_layout(title="Interactive Confusion Matrix")
        fig.show()
    
    def plot_prediction_confidence(self, y_pred_proba, y_true, figsize=(12, 5)):
        """Plot prediction confidence distribution."""
        max_proba = np.max(y_pred_proba, axis=1)
        correct_predictions = (np.argmax(y_pred_proba, axis=1) == y_true)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Confidence distribution
        ax1.hist(max_proba[correct_predictions], bins=30, alpha=0.7, 
                label='Correct Predictions', color='green')
        ax1.hist(max_proba[~correct_predictions], bins=30, alpha=0.7, 
                label='Incorrect Predictions', color='red')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.legend()
        
        # Confidence vs Accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_proba >= confidence_bins[i]) & (max_proba < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                accuracy = np.mean(correct_predictions[mask])
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        ax2.plot(bin_centers, accuracies, 'o-', label='Actual Accuracy')
        ax2.plot([0, 1], [0, 1], '--', label='Perfect Calibration', color='gray')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Calibration Plot')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()