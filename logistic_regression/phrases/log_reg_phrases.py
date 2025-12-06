import pandas as pd
from pathlib import Path
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def lr_phrases(df, tokenized_texts):
    
    x = [" ".join(sent) for sent in tokenized_texts]
    y = df["sentiment"]
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, random_state = 42
    )
    
    vectorizer = CountVectorizer(min_df=3)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    
    model = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced")
    model.fit(x_train_vec, y_train)
    
    y_pred = model.predict(x_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    return acc, report, model


def replace_phrases_in_texts(texts, phrases):

    trie = {}
    for ph in phrases:
        seq = ph.replace("_", " ").split()
        node = trie
        for tok in seq:
            if tok not in node:
                node[tok] = {}
            node = node[tok]
        node["_P"] = ph

    out_texts = []
    for t in texts:
        tokens = word_tokenize((t or "").lower())
        i = 0
        new_tokens = []
        while i < len(tokens):
            node = trie
            j = i
            longest = None
            longest_j = None
            while j < len(tokens) and tokens[j] in node:
                node = node[tokens[j]]
                j += 1
                if "_P" in node:
                    longest = node["_P"]
                    longest_j = j
            if longest is not None:
                new_tokens.append(longest)
                i = longest_j
            else:
                new_tokens.append(tokens[i])
                i += 1
        out_texts.append(" ".join(new_tokens))
        
    return out_texts


if __name__ == "__main__":
    cwd = Path.cwd()

    data_path = cwd / "logistic_regression" / "data" / "amazon_reviews.csv"
    phrases_path = cwd / "logistic_regression" / "phrases" / "phrases_sorted.csv"

    data = pd.read_csv(data_path, encoding="utf-8")
    df = data[["sentiment", "title", "text"]].copy()

    df["sentiment"] = df["sentiment"].replace(1, -1)
    df["sentiment"] = df["sentiment"].replace(2, 1)

    df["text"] = df["text"].fillna("").astype(str)

    phrases_df = pd.read_csv(phrases_path, encoding="utf-8")
    phrases = phrases_df["phrase"].astype(str).tolist()

    texts = df["text"].tolist()
    texts_with_phrases = replace_phrases_in_texts(texts, phrases)

    tokenized_tri = [
        [tok for tok in word_tokenize(t) if (tok.isalpha() or "_" in tok)]
        for t in texts_with_phrases
    ]

    acc, report, model = lr_phrases(df, tokenized_tri)
    print("Accuracy", acc)
    print("Classification Report")
    print(report)