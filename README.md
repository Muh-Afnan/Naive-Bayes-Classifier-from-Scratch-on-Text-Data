# Day 20 — Naive Bayes Classifier from Scratch on Text Data

Naive Bayes text classifier built from scratch. Learns word frequency 
statistics per class and applies Bayes theorem to classify new documents. 
No sklearn, no NLTK.

## What It Does

- Learns class priors and word likelihoods from training data
- Laplace smoothing for unseen words
- Log probability scoring — no underflow
- Three visualizations: word probabilities, class distribution, confusion matrix

## Project Structure

day-20-naive-bayes/
├── src/
│ ├── naive_bayes.py # NaiveBayes class
│ └── visualizer.py # Three visualization functions
├── tests/
│ └── test.py # 8 tests
├── demo.py
├── problem_statement.md
├── approach.md
├── learnings.md
└── README.md


## Quick Start

```bash
python demo.py
```

## Core Class

```python
from src.naive_bayes import NaiveBayes

nb = NaiveBayes(smoothing=1.0)
nb.fit(
    ["win free money", "free prize click", "meeting tomorrow", "lunch today"],
    ["spam", "spam", "not_spam", "not_spam"]
)

print(nb.predict(["win free money"]))        # ["spam"]
print(nb.predict(["meeting tomorrow"]))      # ["not_spam"]
print(nb.predict_proba(["win free money"]))  # [{"spam": 0.89, "not_spam": 0.11}]
```

## Tests

```bash
python -m pytest tests/test.py
```

8 passed in 0.13s


| Test | What It Verifies |
|---|---|
| `test_predict_spam` | Spam email → spam label |
| `test_predict_not_spam` | Ham email → not_spam label |
| `test_predict_proba_sums_to_one` | Probabilities sum to 1.0 |
| `test_unseen_word_no_crash` | Laplace smoothing handles new words |
| `test_tokenize_lowercase` | Output is all lowercase |
| `test_tokenize_removes_punctuation` | No punctuation in output |
| `test_fit_class_counts` | Correct document counts per class |
| `test_fit_vocab` | Vocabulary contains all unique words |

## Math

**Training:**

P(class) = class_doc_count / total_docs
P(word|class) = (word_count + α) / (total_words + α × |vocab|)


**Scoring:**

log P(class|doc) = log P(class) + Σ log P(word|class)


**Classification:**

predict(doc) = argmax_class log P(class|doc)


## Key Insight

Naive Bayes is Bayes theorem (Day 8) applied once per word. The "naive" 
independence assumption — each word contributes independently — is clearly 
wrong linguistically but works because the relative ordering of class scores 
is usually preserved even when absolute probabilities are wrong.

## Dependencies

- `math` — log2 from standard library
- `matplotlib` — visualizations only
- No numpy, no sklearn, no NLTK