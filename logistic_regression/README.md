## Installing Dependencies

From the root directory, run the following commands:

```
pip install -r logistic_regression/requirements.txt
```

```
python -m spacy download en_core_web_sm
```

```
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

# Logistic Regression Results: Words vs Phrases

## **1. Logistic Regression on Words**

**Accuracy:** 0.8362

**Classification Report:**

| Sentiment | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| -1       | 0.84      | 0.83   | 0.83     | 9973    |
| 1        | 0.83      | 0.84   | 0.84     | 10027   |
| **Accuracy** |       |        | 0.84 | 20000 |
| **Macro avg** | 0.84 | 0.84   | 0.84     | 20000   |
| **Weighted avg** | 0.84 | 0.84 | 0.84     | 20000   |

---

## **2. Logistic Regression on Phrases**

**Accuracy:** 0.86245

**Classification Report:**

| Sentiment | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| -1       | 0.8629    | 0.8609 | 0.8619   | 9973    |
| 1        | 0.8620    | 0.8640 | 0.8630   | 10027   |
| **Accuracy** |        |       | 0.8625   | 20000   |
| **Macro avg** | 0.8625 | 0.8624 | 0.8624 | 20000   |
| **Weighted avg** | 0.8625 | 0.8625 | 0.8624 | 20000|

---

# Summary

Using phrases as features:

- Accuracy improves from **0.836 → 0.862**
- Both classes show better precision, recall, and F1