from src.naive_bayes import NaiveBayes
from src.visualizer import plot_word_probabilities, plot_class_distribution, plot_confusion_matrix

# --- Training data ---
X_train = [
    "win free money now click here",
    "free prize you won click",
    "win lottery prize money",
    "buy cheap medicine online",
    "meeting tomorrow at nine am",
    "lunch at noon today",
    "project deadline is friday",
    "call me when you are free",
    "quarterly report due monday",
    "team standup at ten am"
]

y_train = [
    "spam", "spam", "spam", "spam",
    "not_spam", "not_spam", "not_spam", "not_spam", "not_spam", "not_spam"
]

# --- Train ---
nb = NaiveBayes(smoothing=1.0)
nb.fit(X_train, y_train)

print("=== Training Complete ===")
print(f"  Classes:    {nb.classes}")
print(f"  Vocab size: {len(nb.vocab)}")
print(f"  Spam docs:  {nb.class_counts['spam']}")
print(f"  Ham docs:   {nb.class_counts['not_spam']}")

# --- Predict ---
X_test = [
    "win free money",
    "meeting at noon",
    "click here to claim prize",
    "project update tomorrow",
    "free lottery winner"
]
y_test = ["spam", "not_spam", "spam", "not_spam", "spam"]

predictions = nb.predict(X_test)
probas = nb.predict_proba(X_test)

print("\n=== Predictions ===")
for doc, pred, proba in zip(X_test, predictions, probas):
    print(f"  '{doc}'")
    print(f"    → {pred} (spam: {proba.get('spam', 0):.2%}, not_spam: {proba.get('not_spam', 0):.2%})")

accuracy = sum(p == t for p, t in zip(predictions, y_test)) / len(y_test)
print(f"\n  Accuracy: {accuracy:.0%}")

# --- Visualizations ---
plot_class_distribution(y_train)
plot_word_probabilities(nb, "spam", top_n=10)
plot_word_probabilities(nb, "not_spam", top_n=10)
plot_confusion_matrix(y_test, predictions)