import math
class NaiveBayes:
    def __init__(self, smoothing=1.0):
        # smoothing = Laplace smoothing value
        self.smoothing = smoothing
    
    def fit(self, X: list[str], y: list[str]) -> None:
        self.classes = list(set(y))
        self.class_counts = {}
        self.word_counts = {}
        self.class_word_totals = {}
        self.vocab = set()

        for c in self.classes:
            self.class_counts[c]=0
            self.word_counts[c]={}
            self.class_word_totals[c]=0

        for doc,label in zip(X,y):
            tokens = self._tokenize(doc)
            self.class_counts[label] += 1
            for token in tokens:
                self.vocab.add(token)
                self.word_counts[label][token] = self.word_counts[label].get(token,0)+1
                self.class_word_totals[label] += 1
        self.n_docs = len(y)
    
    def predict(self, X: list[str]) -> list[str]:
        result = []
        for doc in X:
            tokens = self._tokenize(doc)
            scores = self._score(tokens)
            best_label = max(scores, key=scores.get)
            result.append(best_label)
        return result

    def predict_proba(self, X: list[str]) -> list[dict]:
        for doc in X:
            tokens = self._tokenize(doc)
            scores = self._score(tokens)
            for score in scores:
                scores.get()
            best_label = max(scores, key=scores.get)

    
    
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        cleaned = ""
        for char in text:
            if char.isalpha() or char == " ":
                cleaned += char
        return [word for word in cleaned.split() if word]

    
    def _score(self, tokens: list[str]) -> dict:
        result = {}
        for label in self.classes:
            class_prob = math.log2(self.class_counts[label]/self.n_docs)
            sum_word_prob = 0
            for token in tokens:
                word_count = self.word_counts[label].get(token, 0) + self.smoothing
                log_p = math.log2(word_count / (self.class_word_totals[label] + self.smoothing * len(self.vocab)))
                sum_word_prob +=log_p
            log_p_class_doc = class_prob + sum_word_prob
            result[label] = log_p_class_doc
        return result


if __name__ == "__main__":
    nb = NaiveBayes()
    nb.fit(
        ["win free money", "free prize click", "meeting tomorrow", "lunch today"],
        ["spam", "spam", "not_spam", "not_spam"]
    )
    scores = nb._score(["win", "free"])
    print(scores)  # spam score should be higher than not_spam