"""
Quick demo script for the Text Classification System.
Uses the built-in movie reviews dataset for faster execution.
"""

import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import TextPreprocessor, FeatureExtractor
from src.models import TextClassifier
from src.evaluation import ModelEvaluator
from src.visualization import TextVisualization
from data.dataset_loader import load_dataset

def main():
    print("="*60)
    print("TEXT CLASSIFICATION SYSTEM - QUICK DEMO")
    print("Ardentix AI/ML Engineer Intern Assignment")
    print("="*60)
    
    # Load movie reviews dataset (small and fast)
    print("\n1. Loading movie reviews dataset...")
    texts, labels, label_names = load_dataset('movie_reviews')
    print(f"✓ Loaded {len(texts)} reviews with {len(label_names)} classes")
    
    # Initialize components
    print("\n2. Initializing components...")
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    evaluator = ModelEvaluator()
    print("✓ Components initialized")
    
    # Preprocess texts
    print("\n3. Preprocessing texts...")
    processed_texts = [preprocessor.preprocess_text(text) for text in texts]
    print("✓ Text preprocessing completed")
    
    # Show preprocessing example
    print("\nPreprocessing Example:")
    print("Original:", texts[0])
    print("Processed:", processed_texts[0])
    
    # Extract features
    print("\n4. Extracting TF-IDF features...")
    X_features = feature_extractor.extract_tfidf_features(processed_texts, max_features=100)
    print(f"✓ Feature matrix shape: {X_features.shape}")
    
    # Train models
    print("\n5. Training models...")
    models_to_test = ['naive_bayes', 'logistic_regression']
    
    for model_type in models_to_test:
        print(f"\nTraining {model_type}...")
        
        # Train model
        classifier = TextClassifier(model_type)
        classifier.train(X_features, labels)
        
        # Evaluate model (using same data for demo purposes)
        metrics = evaluator.evaluate_model(
            classifier, X_features, labels, model_type
        )
        
        print(f"✓ {model_type} - Accuracy: {metrics['accuracy']:.3f}, F1-Score: {metrics['f1_score']:.3f}")
    
    # Model comparison
    print("\n6. Model Comparison:")
    comparison_df = evaluator.compare_models()
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score']].round(3))
    
    # Best model
    best_model_name, best_score = evaluator.get_best_model('f1_score')
    print(f"\n✓ Best Model: {best_model_name} (F1-Score: {best_score:.3f})")
    
    # Test prediction
    print("\n7. Testing prediction...")
    test_text = "This movie was absolutely fantastic! Great acting and storyline."
    processed_test = preprocessor.preprocess_text(test_text)
    test_features = feature_extractor.transform_tfidf([processed_test])
    
    # Get best classifier
    best_result = evaluator.results[best_model_name]
    
    # Make prediction (we need to retrain to get the classifier object)
    best_classifier = TextClassifier(best_model_name)
    best_classifier.train(X_features, labels)
    
    prediction = best_classifier.predict(test_features)[0]
    probabilities = best_classifier.predict_proba(test_features)[0]
    
    print(f"Test text: '{test_text}'")
    print(f"Predicted class: {label_names[prediction]}")
    print(f"Confidence: {probabilities[prediction]:.2%}")
    print(f"All probabilities: {dict(zip(label_names.values(), probabilities))}")
    
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print("✓ Successfully demonstrated complete ML pipeline")
    print("✓ Text preprocessing with cleaning and tokenization")
    print("✓ TF-IDF feature extraction")
    print("✓ Multiple model training and evaluation")
    print("✓ Model comparison and selection")
    print("✓ Real-time text classification")
    
    print("\nKey Features Demonstrated:")
    print("• Comprehensive text preprocessing")
    print("• Feature extraction with TF-IDF")
    print("• Multiple ML algorithms (Naive Bayes, Logistic Regression)")
    print("• Model evaluation with multiple metrics")
    print("• Best model selection")
    print("• Real-time prediction with confidence scores")
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print("For the full system, run: python main.py")
    print("For web interface, run: streamlit run app.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()