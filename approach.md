# Approach

## Core Formula

P(class | doc) ∝ P(class) × Π P(word | class)


In log space (to prevent underflow):

log P(class | doc) = log P(class) + Σ log P(word | class)


## fit — Counting Phase

No probabilities computed during training. Only counts:

- `class_counts` — how many documents per class
- `word_counts` — how many times each word appears per class  
- `class_word_totals` — total words per class
- `vocab` — all unique words across all documents

Probabilities are computed on demand in `_score`.

## _score — Scoring Phase

For each class:

log P(class) = log(class_doc_count / total_docs)

log P(word|class) = log((word_count + smoothing) /
(total_words_in_class + smoothing × vocab_size))


Returns dict of log scores per class.

## Laplace Smoothing

If a word never appeared in training for a class, its count is 0. 
`log(0) = -infinity` — the whole score collapses.

Fix: add `smoothing=1` to every word count:

P(word|class) = (count + 1) / (total + vocab_size)


Now unseen words get a small but non-zero probability.

## predict_proba — Converting Log Scores to Probabilities
Subtract max score (numerical stability)
Exponentiate: 2^score
Normalize: divide by sum

## Tokenizer

Lowercases text, keeps only alphabetic characters and spaces, splits on 
whitespace. Removes all punctuation without external libraries.

## Three Visualizations

**`plot_word_probabilities`** — top N words by frequency for a given class. 
Shows which words are most characteristic of spam vs not spam.

**`plot_class_distribution`** — bar chart of training label counts. Shows 
class balance.

**`plot_confusion_matrix`** — 2×2 grid of TP, FP, TN, FN. Shows where the 
model makes mistakes.