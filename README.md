# IMDB-Movie-Sentiment-Analysis
Sentiment analysis of IMDB movie reviews using natural language processing (NLP) and machine learning techniques. The project involves stemming, lemmatization, TF-IDF vectorization, n-gram feature extraction, and classification using Logistic Regression and LinearSVC.

# Dataset
- Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

- Rows/Columns: ~50,000 records × 2 features  

- Contains data from IMDB, with features including movie reviews (text) and corresponding sentiment labels (positive or negative).

# Tools & Technologies 

- **Python** (Pandas, NumPy)

- **Natural Language Processing**: NLTK 

- **Feature Extraction**: TF-IDF

- **Machine Learning**: Scikit-learn (Logistic Regression, Linear SVC)

- **Model Evaluation**: Accuracy, Confusion Matrix, Classification Report

# Model Development

**Text Preprocessing:**

- Cleaned the raw movie reviews by removing punctuation, special characters, and converting all text to lowercase.

- Applied stemming and lemmatization to normalize words and reduce noise.

- Removed common stopwords to focus on meaningful terms.

**Feature Extraction:**

- Transformed the processed text data into numerical format using TF-IDF vectorization, which helps assign importance to words based on their frequency and uniqueness across documents.

**Model Building:**

Trained and compared two machine learning classification models:

- Logistic Regression

- Linear Support Vector Classifier (LinearSVC)

Used an 80:20 train-test split for model evaluation.

# Evaluation Metrics:

Assessed model performance using:

- Accuracy score

- Confusion matrix

- Classification report (Precision, Recall, F1 Score)

# Observations

- LinearSVC slightly outperformed Logistic Regression with unigrams.

- Accuracy remained high (~89–90%) across both models.
