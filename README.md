# Spam / Ham SMS Classifier

A machine learning model that classifies SMS messages as **spam** or **ham** (not spam), built with Python and scikit-learn.

**Pipeline:** Text cleaning → TF-IDF vectorizer (unigrams + bigrams) → Multinomial Naive Bayes

---

## 1. What you need

| Requirement | Why |
|---|---|
| Python 3.9+ | Language |
| pandas | Load and clean the dataset |
| scikit-learn | TF-IDF, Naive Bayes, metrics |
| matplotlib | Confusion matrix plot |
| joblib | Save/load the trained model |
| **Dataset** | [SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) → download `spam.csv` |
| Git + GitHub account | To upload the project |

## 2. Setup

```bash
git clone https://github.com/<your-username>/spam-ham-classifier.git
cd spam-ham-classifier
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Place `spam.csv` in the project root (same folder as `spam_classifier.py`).

## 3. Run

```bash
python spam_classifier.py                                   # train + evaluate + demo
python spam_classifier.py --predict "You won a free prize!" # classify one message
```

## 4. How it works

1. **Load data**: `v1` = label, `v2` = message. Duplicates and nulls removed.
2. **Clean text**: lowercase, replace URLs and numbers with tokens, strip punctuation.
3. **Split**: 80% train / 20% test, stratified (the dataset is imbalanced, about 87% ham / 13% spam).
4. **Vectorize**: TF-IDF turns words into weighted numeric features.
5. **Train**: Multinomial Naive Bayes, which works well for text classification.
6. **Evaluate**: accuracy, precision, recall, F1, confusion matrix. For spam, **precision matters most**, because wrongly blocking a real message is costly.
7. **Save**: model stored as `spam_model.joblib`.

## 5. Screenshots

### Program code
![Code](screenshots/code.png)

### Training output (metrics)
![Output](screenshots/output.png)

### Confusion matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### Sample predictions
![Predictions](screenshots/predictions.png)

## 6. Results

| Metric | Score |
|---|---|
| Accuracy | _fill in from your output_ |
| Precision | _fill in_ |
| Recall | _fill in_ |
| F1-score | _fill in_ |

## 7. Future improvements

- Compare with Logistic Regression, SVM, Random Forest
- Hyperparameter tuning with GridSearchCV
- Deploy as a REST API (Flask, or Spring Boot calling the model)
- Add a simple web UI

## Tech stack
Python · pandas · scikit-learn · matplotlib · joblib
