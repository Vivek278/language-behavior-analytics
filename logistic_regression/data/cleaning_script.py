import pandas as pd
import spacy
from pathlib import Path

cwd = Path.cwd()
data_path = cwd / "logistic_regression" / "data" / "amazon_reviews.csv"
save_path = cwd / "logistic_regression" / "data" / "clean_amazon_reviews.csv"

df = pd.read_csv(data_path, encoding="utf-8")

nlp = spacy.load("en_core_web_sm")
texts = df["text"].astype(str).str.lower().tolist()

clean_texts = []

for doc in nlp.pipe(texts, batch_size=1000):
    clean_texts.append(
        " ".join(
            token.lemma_
            for token in doc
            if not token.is_stop and token.is_alpha
        )
    )

df["clean_text"] = clean_texts

df.to_csv(save_path, index=False, encoding="utf-8")