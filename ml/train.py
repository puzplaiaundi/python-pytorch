import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

DATASET_PATH = Path(__file__).with_name("dataset.csv")
MODEL_PATH = Path(__file__).with_name("model.joblib")

def main():
    df = pd.read_csv(DATASET_PATH)
    print("Leyendo dataset desde:", DATASET_PATH)
    print(df["label"].value_counts())

    X = df["texto"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()
