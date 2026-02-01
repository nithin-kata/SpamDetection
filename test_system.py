"""
Quick test script to verify the text classification system works correctly.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    try:
        from src.preprocessing import TextPreprocessor, FeatureExtractor
        from src.models import TextClassifier, ModelTuner
        from src.evaluation import ModelEvaluator
        from src.visualization import TextVisualization
        from data.dataset_loader import load_dataset, get_available_datasets
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_preprocessing():
    """Test text preprocessing functionality."""
    print("\nTesting preprocessing...")
    try:
        from src.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        sample_text = "This is a SAMPLE text with URLs http://example.com and emails test@email.com!"
        processed = preprocessor.preprocess_text(sample_text)
        
        print(f"Original: {sample_text}")
        print(f"Processed: {processed}")
        print("✓ Preprocessing works")
        return True
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction."""
    print("\nTesting feature extraction...")
    try:
        from src.preprocessing import TextPreprocessor, FeatureExtractor
        
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor()
        
        texts = [
            "This is a positive example",
            "This is a negative example",
            "Another positive text sample"
        ]
        
        processed_texts = [preprocessor.preprocess_text(text) for text in texts]
        features = feature_extractor.extract_tfidf_features(processed_texts, max_features=100)
        
        print(f"Feature matrix shape: {features.shape}")
        print("✓ Feature extraction works")
        return True
    except Exception as e:
        print(f"✗ Feature extraction error: {e}")
        return False

def test_model_training():
    """Test model training."""
    print("\nTesting model training...")
    try:
        from src.preprocessing import TextPreprocessor, FeatureExtractor
        from src.models import TextClassifier
        
        # Simple test data
        texts = [
            "positive example one", "positive example two", "positive sample",
            "negative example one", "negative example two", "negative sample"
        ]
        labels = [1, 1, 1, 0, 0, 0]
        
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor()
        
        processed_texts = [preprocessor.preprocess_text(text) for text in texts]
        features = feature_extractor.extract_tfidf_features(processed_texts, max_features=50)
        
        # Train model
        classifier = TextClassifier('naive_bayes')
        classifier.train(features, labels)
        
        # Test prediction
        predictions = classifier.predict(features)
        print(f"Predictions: {predictions}")
        print("✓ Model training works")
        return True
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False

def test_evaluation():
    """Test model evaluation."""
    print("\nTesting evaluation...")
    try:
        from src.preprocessing import TextPreprocessor, FeatureExtractor
        from src.models import TextClassifier
        from src.evaluation import ModelEvaluator
        
        # Simple test data
        texts = [
            "positive example one", "positive example two", "positive sample",
            "negative example one", "negative example two", "negative sample"
        ]
        labels = [1, 1, 1, 0, 0, 0]
        
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor()
        evaluator = ModelEvaluator()
        
        processed_texts = [preprocessor.preprocess_text(text) for text in texts]
        features = feature_extractor.extract_tfidf_features(processed_texts, max_features=50)
        
        # Train and evaluate
        classifier = TextClassifier('naive_bayes')
        classifier.train(features, labels)
        
        metrics = evaluator.evaluate_model(classifier, features, labels, "test_model")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print("✓ Evaluation works")
        return True
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    try:
        from data.dataset_loader import get_available_datasets, load_dataset
        
        datasets = get_available_datasets()
        print(f"Available datasets: {list(datasets.keys())}")
        
        # Test movie reviews (smallest dataset)
        texts, labels, label_names = load_dataset('movie_reviews')
        print(f"Loaded {len(texts)} texts with {len(label_names)} classes")
        print("✓ Dataset loading works")
        return True
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("TEXT CLASSIFICATION SYSTEM - QUICK TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_preprocessing,
        test_feature_extraction,
        test_model_training,
        test_evaluation,
        test_dataset_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main.py' for the complete pipeline")
        print("2. Run 'streamlit run app.py' for the web interface")
        print("3. Open the Jupyter notebook for interactive exploration")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()