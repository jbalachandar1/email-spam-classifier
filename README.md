# email-spam-classifier

A simple Natural Language Processing (NLP) project that builds a binary classifier to detect spam versus ham (legitimate) emails. Leveraging TF–IDF vectorization and Logistic Regression, this model achieves 100% accuracy on a held-out test set.

## Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Training and Evaluation](#training-and-evaluation)  
  - [Manual Testing](#manual-testing)  
- [Results](#results)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  


## Features

- TF–IDF vectorization with English stop-word removal and bi-grams  
- Train/test split with reproducible random seed  
- Logistic Regression (`solver='liblinear'`, `max_iter=1000`)  
- Classification report (precision, recall, f1-score)  
- Utility function for manual inference on new email text  

## Dataset

- **File:** `spam_mail_classifier.csv`  
- **Columns:**  
  - `email_text` (string)  
  - `label` (`ham` or `spam`)  


## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/spam-email-classifier.git
   cd spam-email-classifier


2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate.bat      # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training and Evaluation

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('spam_mail_classifier.csv')
X_text, y = df['email_text'], df['label']

# Vectorize
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2
)
X = vectorizer.fit_transform(X_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham','spam']))
```

### Manual Testing

```python
def predict_spam(email_text: str) -> str:
    """Return 'spam' or 'ham' for new email text."""
    X_new = vectorizer.transform([email_text])
    return model.predict(X_new)[0]

# Example
sample_email = "Hey, want to get rich quick? Click here..."
print(predict_spam(sample_email))  # spam
```

## Results

```text
              precision    recall  f1-score   support
         ham       1.00      1.00      1.00       122
        spam       1.00      1.00      1.00        78
    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
```

## Project Structure

```
spam-email-classifier/
├── data/
│   └── spam_mail_classifier.csv
├── docs/
│   └── confusion_matrix.png
├── src/
│   └── classifier.py      # main training & inference code
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Commit your changes (`git commit -m 'Add feature xyz'`)
4. Push to your branch (`git push origin feature/xyz`)
5. Open a pull request

```
```
