"""
Main script for the Text Classification System.
Demonstrates the complete ML pipeline from data loading to model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.preprocessing import TextPreprocessor, FeatureExtractor
from src.models import TextClassifier, ModelTuner
from src.evaluation import ModelEvaluator
from src.visualization import TextVisualization

def load_sample_data():
    """Load sample dataset for demonstration."""
    print("Loading sample dataset...")
    
    # Using 20 newsgroups dataset as an example
    # You can replace this with your own dataset
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )
    
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Combine train and test for our own split
    texts = list(newsgroups_train.data) + list(newsgroups_test.data)
    labels = list(newsgroups_train.target) + list(newsgroups_test.target)
    
    # Create label names mapping
    label_names = {i: name for i, name in enumerate(newsgroups_train.target_names)}
    
    print(f"Loaded {len(texts)} documents across {len(categories)} categories")
    return texts, labels, label_names

def main():
    """Main execution pipeline."""
    print("="*60)
    print("TEXT CLASSIFICATION SYSTEM")
    print("Ardentix AI/ML Engineer Intern Assignment")
    print("="*60)
    
    # 1. Load Data
    texts, labels, label_names = load_sample_data()
    
    # 2. Initialize components
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    evaluator = ModelEvaluator()
    visualizer = TextVisualization()
    
    # 3. Data Exploration and Visualization
    print("\n" + "="*40)
    print("DATA EXPLORATION")
    print("="*40)
    
    print(f"Dataset size: {len(texts)} documents")
    print(f"Number of classes: {len(set(labels))}")
    print(f"Classes: {list(label_names.values())}")
    
    # Visualize class distribution
    class_counts = visualizer.plot_class_distribution(
        [label_names[label] for label in labels],
        title="Dataset Class Distribution"
    )
    
    # 4. Text Preprocessing
    print("\n" + "="*40)
    print("TEXT PREPROCESSING")
    print("="*40)
    
    print("Preprocessing text data...")
    processed_texts = [preprocessor.preprocess_text(text) for text in texts]
    
    # Show preprocessing example
    print("\nPreprocessing Example:")
    print("Original:", texts[0][:200] + "...")
    print("Processed:", processed_texts[0][:200] + "...")
    
    # Visualize text length distribution
    visualizer.plot_text_length_distribution(
        processed_texts, 
        [label_names[label] for label in labels]
    )
    
    # 5. Feature Extraction
    print("\n" + "="*40)
    print("FEATURE EXTRACTION")
    print("="*40)
    
    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("Extracting TF-IDF features...")
    X_train_tfidf = feature_extractor.extract_tfidf_features(X_train_text)
    X_test_tfidf = feature_extractor.transform_tfidf(X_test_text)
    
    print("Extracting Bag of Words features...")
    X_train_bow = feature_extractor.extract_bow_features(X_train_text)
    X_test_bow = feature_extractor.transform_bow(X_test_text)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    print(f"BoW feature matrix shape: {X_train_bow.shape}")
    
    # 6. Model Training and Evaluation
    print("\n" + "="*40)
    print("MODEL TRAINING & EVALUATION")
    print("="*40)
    
    # Define models to test
    models_to_test = ['naive_bayes', 'logistic_regression']
    feature_sets = [
        ('TF-IDF', X_train_tfidf, X_test_tfidf),
        ('BoW', X_train_bow, X_test_bow)
    ]
    
    best_model_info = {'score': 0, 'model': None, 'name': None}
    
    for feature_name, X_train_feat, X_test_feat in feature_sets:
        print(f"\n--- Testing with {feature_name} features ---")
        
        for model_type in models_to_test:
            model_name = f"{model_type}_{feature_name}"
            print(f"\nTraining {model_name}...")
            
            # Train model
            classifier = TextClassifier(model_type)
            classifier.train(X_train_feat, y_train)
            
            # Evaluate model
            metrics = evaluator.evaluate_model(
                classifier, X_test_feat, y_test, model_name
            )
            
            # Print results
            evaluator.print_evaluation_report(model_name)
            
            # Track best model
            if metrics['f1_score'] > best_model_info['score']:
                best_model_info.update({
                    'score': metrics['f1_score'],
                    'model': classifier,
                    'name': model_name
                })
    
    # 7. Model Comparison and Visualization
    print("\n" + "="*40)
    print("MODEL COMPARISON")
    print("="*40)
    
    # Compare all models
    comparison_df = evaluator.compare_models()
    print("\nModel Comparison Summary:")
    print(comparison_df.round(4))
    
    # Plot model comparison
    evaluator.plot_model_comparison()
    
    # Best model analysis
    best_model_name, best_score = evaluator.get_best_model('f1_score')
    print(f"\nBest Model: {best_model_name} (F1-Score: {best_score:.4f})")
    
    # Plot confusion matrix for best model
    evaluator.plot_confusion_matrix(
        best_model_name, 
        class_names=list(label_names.values())
    )
    
    # 8. Feature Analysis
    print("\n" + "="*40)
    print("FEATURE ANALYSIS")
    print("="*40)
    
    # Get feature importance from best model
    if 'tfidf' in best_model_name.lower():
        feature_names = feature_extractor.tfidf_vectorizer.get_feature_names_out()
    else:
        feature_names = feature_extractor.count_vectorizer.get_feature_names_out()
    
    feature_importance = best_model_info['model'].get_feature_importance(feature_names)
    if feature_importance:
        visualizer.plot_feature_importance(feature_importance, top_n=20)
    
    # 9. Word Clouds
    print("\n" + "="*40)
    print("WORD CLOUD VISUALIZATION")
    print("="*40)
    
    # Create word clouds for each class
    for label_idx, class_name in label_names.items():
        visualizer.create_wordcloud(
            processed_texts, 
            labels, 
            class_name=label_idx,
            figsize=(12, 6)
        )
    
    # 10. Summary and Recommendations
    print("\n" + "="*60)
    print("SUMMARY AND OBSERVATIONS")
    print("="*60)
    
    print(f"✓ Successfully processed {len(texts)} documents")
    print(f"✓ Tested {len(models_to_test)} different algorithms")
    print(f"✓ Compared {len(feature_sets)} feature extraction methods")
    print(f"✓ Best performing model: {best_model_name}")
    print(f"✓ Best F1-Score achieved: {best_score:.4f}")
    
    print("\nKey Observations:")
    print("• TF-IDF generally performs better than Bag of Words for text classification")
    print("• Logistic Regression often provides good performance with proper regularization")
    print("• Naive Bayes is fast and works well with limited training data")
    print("• Text preprocessing significantly impacts model performance")
    
    print("\nRecommendations for improvement:")
    print("• Try advanced models like SVM or Random Forest")
    print("• Experiment with different n-gram ranges")
    print("• Consider using word embeddings (Word2Vec, GloVe)")
    print("• Implement cross-validation for more robust evaluation")
    print("• Add more sophisticated text preprocessing steps")
    
    print(f"\n{'='*60}")
    print("Analysis complete! Check the Streamlit app for interactive exploration.")
    print("Run: streamlit run app.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()