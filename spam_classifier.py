"""
Spam / Ham SMS classifier
Pipeline: clean text -> TF-IDF -> Multinomial Naive Bayes
Usage:
    python spam_classifier.py            # train + evaluate + demo
    python spam_classifier.py --predict "Congratulations! You won a free prize"
"""
import argparse
import re
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = "spam.csv"      # Kaggle "SMS Spam Collection" (columns: v1, v2)
MODEL_PATH = "spam_model.joblib"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)   # links
    text = re.sub(r"\d+", " num ", text)              # numbers
    text = re.sub(r"[^a-z\s]", " ", text)             # punctuation/symbols
    return re.sub(r"\s+", " ", text).strip()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df = df.drop_duplicates().dropna()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean"] = df["message"].apply(clean_text)
    return df


def train():
    df = load_data(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df["label"].value_counts(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42,
        stratify=df["label"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("nb", MultinomialNB(alpha=0.1)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall   : {recall_score(y_test, pred):.4f}")
    print(f"F1-score : {f1_score(y_test, pred):.4f}\n")
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["ham", "spam"], cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("screenshots/confusion_matrix.png", dpi=150, bbox_inches="tight")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def predict(model, message: str) -> str:
    label = model.predict([clean_text(message)])[0]
    prob = model.predict_proba([clean_text(message)])[0][label]
    return f"{'SPAM' if label else 'HAM'} ({prob:.1%} confident)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=str, help="message to classify")
    args = parser.parse_args()

    if args.predict:
        try:
            model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            sys.exit("No saved model. Run `python spam_classifier.py` first.")
        print(predict(model, args.predict))
    else:
        model = train()
        print("\n--- Demo predictions ---")
        for msg in ["Congratulations! You won a free iPhone. Click http://win.com now",
                    "Hey, are we still meeting for lunch tomorrow?",
                    "URGENT! Your account is blocked. Call 09061234567 to claim"]:
            print(f"{msg[:55]:<57} -> {predict(model, msg)}")
