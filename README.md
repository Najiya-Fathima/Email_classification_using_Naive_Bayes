# Email Spam Classifier

A machine learning project that classifies emails as spam or ham (legitimate) using Naive Bayes algorithm with natural language processing techniques.

## Overview

This project implements an email spam detection system that processes raw email data, extracts meaningful features through text preprocessing, and uses a Multinomial Naive Bayes classifier to distinguish between spam and legitimate emails.

## Features

- **Email Text Preprocessing**: Custom transformer that handles multipart emails, HTML parsing, and text cleaning
- **Advanced Text Processing**: Includes stemming, lemmatization, stopword removal, and feature extraction
- **Machine Learning Pipeline**: Scikit-learn pipeline for streamlined preprocessing and classification
- **Model Persistence**: Saves trained model using joblib for future use
- **Performance Metrics**: Accuracy, F1-score, and confusion matrix visualization

## Dataset Structure

The project expects the following directory structure:
```
dataset/
└── archieve/
    ├── easy_ham/
    │   └── easy_ham/
    ├── hard_ham/
    │   └── hard_ham/
    └── spam_2/
        └── spam_2/
```

## Dependencies

```python
numpy
pandas
matplotlib
seaborn
beautifulsoup4
nltk
scikit-learn
joblib
```

### NLTK Data Requirements
Download the following NLTK data:
```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn beautifulsoup4 nltk scikit-learn joblib
   ```
3. Download NLTK data as mentioned above
4. Ensure your dataset follows the expected directory structure

## Usage

### Running the Classification

```python
python train_model.py
```

The script will:
1. Load email data from the specified directories
2. Preprocess the text data
3. Train a Naive Bayes classifier
4. Evaluate the model performance
5. Save the trained pipeline

### Using the Saved Model

```python
import joblib

# Load the trained pipeline
pipeline = joblib.load("email_classifier_naive.joblib")

# Classify new emails
new_emails = ["Your email content here..."]
predictions = pipeline.predict(new_emails)
```

## Text Preprocessing Pipeline

The `email_to_clean_text` transformer performs the following operations:

1. **Email Parsing**: Extracts text content from email headers and multipart messages
2. **HTML Processing**: Uses BeautifulSoup to extract plain text from HTML content
3. **Text Cleaning**:
   - Converts to lowercase
   - Removes URLs and email addresses
   - Removes punctuation and digits
   - Removes English stopwords
4. **Text Normalization**:
   - Lemmatization using WordNet
   - Stemming using Porter Stemmer

## Model Architecture

- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: Count Vectorization
- **Text Processing**: Custom email-to-text transformer
- **Pipeline Structure**: Text Processing → Vectorization → Classification

## Performance Metrics

The model outputs:
- **Accuracy Score**: Overall classification accuracy - 95.6%
- **F1 Score**: Harmonic mean of precision and recall - 0.93
- **Confusion Matrix**: Visual representation of classification results

## File Structure

```
├── train_model.py          # Main classification script
├── email_classifier_naive.joblib # Saved trained model
├── dataset/                     # Email dataset directory
│   └── archieve/
│       ├── easy_ham/
│       ├── hard_ham/
│       └── spam_2/
├── check_email.py               #Checking with other emails
└── README.md                    # This file
```

## Key Components

### Custom Transformer: `email_to_clean_text`
- Inherits from `BaseEstimator` and `TransformerMixin`
- Handles email parsing and text preprocessing
- Compatible with scikit-learn pipelines

### Data Processing
- Combines easy_ham and hard_ham as legitimate emails
- Shuffles data to prevent ordering bias
- Uses stratified train-test split (80-20)

### Model Training
- Uses Count Vectorization for feature extraction
- Trains Multinomial Naive Bayes classifier
- Evaluates using multiple metrics

## Notes

- The script uses ISO-8859-1 encoding for reading email files
- Random seed (49) is set for reproducible results
- The model handles both multipart and plain text emails
- Preprocessing removes common noise (URLs, email addresses, punctuation)

## Future Improvements

- Add support for additional email formats
- Implement cross-validation for better model evaluation
- Experiment with other algorithms (SVM, Random Forest)
- Add feature importance analysis
- Implement real-time email classification API

## License

This project is open source and available under the MIT License.

