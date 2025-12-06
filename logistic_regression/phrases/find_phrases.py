import pandas as pd
import numpy as np
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize

from gensim.models.phrases import Phrases, Phraser
from collections import Counter


def get_phrases(df, min_count = 10, threshold = 10):
    x = df["text"].tolist()

    tokenized = [word_tokenize(t.lower()) for t in x]
    tokenized = [[tok for tok in sent if tok.isalpha()] for sent in tokenized]

    bigram_model = Phrases(tokenized, min_count = min_count, threshold = threshold)
    bigram_phraser = Phraser(bigram_model)
    tokenized_bi = [bigram_phraser[sent] for sent in tokenized]

    trigram_model = Phrases(tokenized_bi, min_count = min_count, threshold = threshold)
    trigram_phraser = Phraser(trigram_model)
    tokenized_tri = [trigram_phraser[sent] for sent in tokenized_bi]

    phrase_counts = Counter()
    for sent in tokenized_tri:
        for tok in sent:
            if "_" in tok:
                phrase_counts[tok] += 1

    phrases_atleast2 = {p: c for p, c in phrase_counts.items() if c >= min_count}
    return phrases_atleast2, tokenized_tri


if __name__ == "__main__":

    cwd = Path.cwd()
    data_path = cwd / "logistic_regression" / "data" / "amazon_reviews.csv"
    save_path = cwd / "logistic_regression" / "phrases" / "phrases_sorted.csv"

    data = pd.read_csv(data_path, encoding="utf-8")
    
    df = data[["text"]].copy()
    df["text"] = df["text"].fillna("")
    df["text"] = df["text"].astype(str)

    phrases_atleast2, tokenized_tri = get_phrases(df)

    print("Top 20 phrases:\n")
    for p, c in Counter(phrases_atleast2).most_common(20):
        print(f"{p}: {c}")
        
    phrases_sorted = sorted(phrases_atleast2.items(), key = lambda x: -x[1])
    pd.DataFrame(phrases_sorted, columns=["phrase", "count"]).to_csv(save_path, index = False)
