"""
Streamlit web application for the Text Classification System.
Provides an interactive interface for text classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import joblib
import os

# Import our custom modules
from src.preprocessing import TextPreprocessor, FeatureExtractor
from src.models import TextClassifier
from src.evaluation import ModelEvaluator
from src.visualization import TextVisualization

# Page configuration
st.set_page_config(
    page_title="Text Classification System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load and cache sample dataset."""
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
    
    texts = list(newsgroups_train.data) + list(newsgroups_test.data)
    labels = list(newsgroups_train.target) + list(newsgroups_test.target)
    label_names = {i: name for i, name in enumerate(newsgroups_train.target_names)}
    
    return texts, labels, label_names

@st.cache_resource
def initialize_components():
    """Initialize and cache preprocessing components."""
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    evaluator = ModelEvaluator()
    visualizer = TextVisualization()
    
    return preprocessor, feature_extractor, evaluator, visualizer

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📝 Text Classification System</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Ardentix AI/ML Engineer Intern Assignment**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Exploration", "🔧 Model Training", "🎯 Prediction", "📈 Results Analysis"]
    )
    
    # Initialize components
    preprocessor, feature_extractor, evaluator, visualizer = initialize_components()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Exploration":
        show_data_exploration(preprocessor, visualizer)
    elif page == "🔧 Model Training":
        show_model_training(preprocessor, feature_extractor, evaluator)
    elif page == "🎯 Prediction":
        show_prediction_page(preprocessor, feature_extractor)
    elif page == "📈 Results Analysis":
        show_results_analysis(evaluator, visualizer)

def show_home_page():
    """Display home page with project overview."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">Project Overview</h2>', 
                   unsafe_allow_html=True)
        
        st.write("""
        This text classification system demonstrates a complete machine learning pipeline
        for categorizing text documents. The system includes:
        """)
        
        features = [
            "🔍 **Text Preprocessing**: Cleaning, tokenization, stopword removal, stemming",
            "📊 **Feature Extraction**: TF-IDF and Bag of Words vectorization",
            "🤖 **Multiple Models**: Naive Bayes, Logistic Regression, and more",
            "📈 **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score",
            "📋 **Interactive Interface**: Real-time predictions and visualizations",
            "🎨 **Rich Visualizations**: Word clouds, confusion matrices, feature importance"
        ]
        
        for feature in features:
            st.markdown(feature)
        
        st.markdown('<h2 class="section-header">Technical Stack</h2>', 
                   unsafe_allow_html=True)
        
        tech_cols = st.columns(3)
        with tech_cols[0]:
            st.write("**Core Libraries:**")
            st.write("• Python 3.8+")
            st.write("• Scikit-learn")
            st.write("• NLTK")
            st.write("• Pandas & NumPy")
        
        with tech_cols[1]:
            st.write("**Visualization:**")
            st.write("• Matplotlib")
            st.write("• Seaborn")
            st.write("• Plotly")
            st.write("• WordCloud")
        
        with tech_cols[2]:
            st.write("**Web Interface:**")
            st.write("• Streamlit")
            st.write("• Interactive widgets")
            st.write("• Real-time processing")
            st.write("• Responsive design")
    
    with col2:
        st.markdown('<h2 class="section-header">Quick Stats</h2>', 
                   unsafe_allow_html=True)
        
        # Load data for stats
        texts, labels, label_names = load_sample_data()
        
        st.metric("Total Documents", len(texts))
        st.metric("Number of Classes", len(label_names))
        st.metric("Average Text Length", f"{np.mean([len(text.split()) for text in texts]):.0f} words")
        
        st.markdown('<h2 class="section-header">Dataset Classes</h2>', 
                   unsafe_allow_html=True)
        
        for i, class_name in label_names.items():
            count = labels.count(i)
            st.write(f"**{class_name}**: {count} documents")

def show_data_exploration(preprocessor, visualizer):
    """Display data exploration page."""
    
    st.markdown('<h2 class="section-header">Data Exploration</h2>', 
               unsafe_allow_html=True)
    
    # Load data
    texts, labels, label_names = load_sample_data()
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", len(texts))
    with col2:
        st.metric("Classes", len(label_names))
    with col3:
        avg_length = np.mean([len(text.split()) for text in texts])
        st.metric("Avg Length", f"{avg_length:.0f} words")
    with col4:
        max_length = max([len(text.split()) for text in texts])
        st.metric("Max Length", f"{max_length} words")
    
    # Class distribution
    st.markdown('<h3 class="section-header">Class Distribution</h3>', 
               unsafe_allow_html=True)
    
    class_counts = pd.Series(labels).value_counts().sort_index()
    class_names = [label_names[i] for i in class_counts.index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    ax1.bar(class_names, class_counts.values, color=sns.color_palette("husl", len(class_names)))
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Pie chart
    ax2.pie(class_counts.values, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Proportion')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Sample documents
    st.markdown('<h3 class="section-header">Sample Documents</h3>', 
               unsafe_allow_html=True)
    
    selected_class = st.selectbox("Select a class to view samples:", 
                                 list(label_names.values()))
    
    class_idx = [k for k, v in label_names.items() if v == selected_class][0]
    class_texts = [text for text, label in zip(texts, labels) if label == class_idx]
    
    sample_idx = st.slider("Sample index:", 0, min(len(class_texts)-1, 10), 0)
    
    st.write("**Original Text:**")
    st.text_area("", class_texts[sample_idx], height=200, disabled=True)
    
    # Preprocessing example
    if st.button("Show Preprocessed Version"):
        processed = preprocessor.preprocess_text(class_texts[sample_idx])
        st.write("**Preprocessed Text:**")
        st.text_area("", processed, height=100, disabled=True)

def show_model_training(preprocessor, feature_extractor, evaluator):
    """Display model training page."""
    
    st.markdown('<h2 class="section-header">Model Training</h2>', 
               unsafe_allow_html=True)
    
    # Load and preprocess data
    texts, labels, label_names = load_sample_data()
    
    with st.spinner("Preprocessing text data..."):
        processed_texts = [preprocessor.preprocess_text(text) for text in texts]
    
    # Training configuration
    st.markdown('<h3 class="section-header">Training Configuration</h3>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model:",
            ["naive_bayes", "logistic_regression"]
        )
        
        feature_type = st.selectbox(
            "Select Features:",
            ["TF-IDF", "Bag of Words"]
        )
    
    with col2:
        test_size = st.slider("Test Size:", 0.1, 0.4, 0.2, 0.05)
        max_features = st.slider("Max Features:", 1000, 10000, 5000, 500)
    
    if st.button("Train Model", type="primary"):
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Feature extraction
        with st.spinner("Extracting features..."):
            if feature_type == "TF-IDF":
                X_train_feat = feature_extractor.extract_tfidf_features(X_train_text, max_features)
                X_test_feat = feature_extractor.transform_tfidf(X_test_text)
            else:
                X_train_feat = feature_extractor.extract_bow_features(X_train_text, max_features)
                X_test_feat = feature_extractor.transform_bow(X_test_text)
        
        # Train model
        with st.spinner("Training model..."):
            classifier = TextClassifier(model_type)
            classifier.train(X_train_feat, y_train)
        
        # Evaluate model
        with st.spinner("Evaluating model..."):
            metrics = evaluator.evaluate_model(
                classifier, X_test_feat, y_test, f"{model_type}_{feature_type}"
            )
        
        # Display results
        st.success("Model training completed!")
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        # Save model and components
        model_name = f"{model_type}_{feature_type}"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save model
        classifier.save_model(f"models/{model_name}_model.pkl")
        joblib.dump(feature_extractor, f"models/{model_name}_features.pkl")
        joblib.dump(label_names, f"models/{model_name}_labels.pkl")
        
        st.session_state['trained_model'] = classifier
        st.session_state['feature_extractor'] = feature_extractor
        st.session_state['label_names'] = label_names
        st.session_state['model_name'] = model_name
        
        st.info(f"Model saved as: {model_name}")

def show_prediction_page(preprocessor, feature_extractor):
    """Display prediction page."""
    
    st.markdown('<h2 class="section-header">Text Classification Prediction</h2>', 
               unsafe_allow_html=True)
    
    # Check if model is trained
    if 'trained_model' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' page.")
        return
    
    # Get trained components
    classifier = st.session_state['trained_model']
    feature_extractor = st.session_state['feature_extractor']
    label_names = st.session_state['label_names']
    
    # Input text
    st.markdown('<h3 class="section-header">Enter Text to Classify</h3>', 
               unsafe_allow_html=True)
    
    input_text = st.text_area(
        "Input Text:",
        placeholder="Enter the text you want to classify...",
        height=150
    )
    
    if st.button("Classify Text", type="primary") and input_text.strip():
        
        # Preprocess input
        processed_input = preprocessor.preprocess_text(input_text)
        
        # Extract features
        if hasattr(feature_extractor, 'tfidf_vectorizer') and feature_extractor.tfidf_vectorizer:
            input_features = feature_extractor.transform_tfidf([processed_input])
        else:
            input_features = feature_extractor.transform_bow([processed_input])
        
        # Make prediction
        prediction = classifier.predict(input_features)[0]
        probabilities = classifier.predict_proba(input_features)[0]
        
        # Display results
        st.markdown('<h3 class="section-header">Prediction Results</h3>', 
                   unsafe_allow_html=True)
        
        predicted_class = label_names[prediction]
        confidence = probabilities[prediction]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Predicted Class:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2%}")
        
        with col2:
            st.write("**All Class Probabilities:**")
            prob_df = pd.DataFrame({
                'Class': [label_names[i] for i in range(len(probabilities))],
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            st.dataframe(prob_df, hide_index=True)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(prob_df['Class'], prob_df['Probability'], 
                     color=['green' if cls == predicted_class else 'lightblue' 
                           for cls in prob_df['Class']])
        ax.set_title('Classification Probabilities')
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.tick_params(axis='x', rotation=45)
        
        # Highlight predicted class
        for i, bar in enumerate(bars):
            if prob_df.iloc[i]['Class'] == predicted_class:
                bar.set_color('green')
                bar.set_alpha(0.8)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show preprocessing steps
        with st.expander("View Preprocessing Steps"):
            st.write("**Original Text:**")
            st.text(input_text)
            st.write("**Preprocessed Text:**")
            st.text(processed_input)

def show_results_analysis(evaluator, visualizer):
    """Display results analysis page."""
    
    st.markdown('<h2 class="section-header">Results Analysis</h2>', 
               unsafe_allow_html=True)
    
    if not evaluator.results:
        st.warning("No model results available. Please train a model first.")
        return
    
    # Model comparison
    st.markdown('<h3 class="section-header">Model Performance Comparison</h3>', 
               unsafe_allow_html=True)
    
    comparison_df = evaluator.compare_models()
    if comparison_df is not None:
        st.dataframe(comparison_df.round(4))
        
        # Best model
        best_model, best_score = evaluator.get_best_model('f1_score')
        st.success(f"**Best Model:** {best_model} (F1-Score: {best_score:.4f})")
    
    # Detailed analysis for selected model
    st.markdown('<h3 class="section-header">Detailed Model Analysis</h3>', 
               unsafe_allow_html=True)
    
    model_names = list(evaluator.results.keys())
    selected_model = st.selectbox("Select model for detailed analysis:", model_names)
    
    if selected_model:
        result = evaluator.results[selected_model]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        metrics = result['metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        # Confusion Matrix
        st.markdown('<h4>Confusion Matrix</h4>')
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(result['y_true'], result['y_pred'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {selected_model}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        st.pyplot(fig)
        
        # Classification Report
        st.markdown('<h4>Classification Report</h4>')
        from sklearn.metrics import classification_report
        report = classification_report(result['y_true'], result['y_pred'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))

if __name__ == "__main__":
    main()