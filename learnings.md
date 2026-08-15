# Learnings

## Key Concepts

**Same math as Day 8, applied to words** — Day 8 updated belief about coin 
bias after seeing flips. Day 20 updates belief about spam after seeing each 
word. The formula is identical — only the evidence changes.

**Naive means independent** — the model treats each word as independent 
evidence. "Win" and "free" together don't get special treatment. Clearly 
wrong linguistically, but works well in practice because the relative 
ordering of class scores is usually preserved.

**Log probabilities prevent underflow** — multiplying 50 small probabilities 
produces a number too small for floating point. Adding their logs is 
mathematically identical but numerically stable.

**Laplace smoothing is essential** — without it, one unseen word makes the 
entire class probability zero. Adding 1 to all counts ensures every word 
has a non-zero probability in every class.

**fit only counts, _score computes** — separating counting from probability 
computation keeps the code clean. Training is fast (one pass through data). 
Scoring is flexible (can change smoothing without retraining).

## Bugs Fixed

- **`self.word_counts.get(token, 0)`** — looked in outer dict instead of 
  class-specific dict; fixed to `self.word_counts[label].get(token, 0)`
- **`self.word_counts[label][token]`** — crashed on unseen words; fixed with 
  `.get(token, 0)` in `_score`
- **`predict` appending score dict** — was appending full scores dict instead 
  of best label; fixed to `max(scores, key=scores.get)`
- **`predict_proba` not normalizing** — raw exponentiated scores don't sum 
  to 1; fixed by dividing by total

## Surprises

- Naive Bayes trained on 4 documents still classifies correctly — it's 
  surprisingly effective with very little data
- "free" appears in both spam and ham — but its relative frequency in spam 
  is much higher, so it still pushes toward spam
- Subtracting max score before exponentiating makes no difference to the 
  final probabilities (normalization cancels it) but prevents overflow
- Tokenizer with no external libraries is just 4 lines — keep it simple