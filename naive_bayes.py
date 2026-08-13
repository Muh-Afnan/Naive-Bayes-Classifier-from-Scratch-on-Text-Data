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
    
    def predict(self, X: list[str]) -> list[str]: ...
    
    def predict_proba(self, X: list[str]) -> list[dict]: ...
    # returns [{"spam": 0.9, "not_spam": 0.1}, ...]
    
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        cleaned = ""
        for char in text:
            if char.isalpha() or char == " ":
                cleaned += char
        return [word for word in cleaned.split() if word]

    
    def _score(self, tokens: list[str]) -> dict:
        self.class_prob = {}
        for single in self.classes:
            self.class_prob[single] = self.class_counts[single]/self.n_docs
        for label in self.classes:
            for token in tokens:
                word_count = self.word_counts[label][token] + self.smoothing
                log_p = word_count / self.class_word_totals[label] + self.smoothing



if __name__ == "__main__":
    nb = NaiveBayes()
    nb.fit(
        ["win free money", "free prize click", "meeting tomorrow", "lunch today"],
        ["spam", "spam", "not_spam", "not_spam"]
    )
    print(nb.class_counts)
    print(nb.word_counts["spam"])
    print(nb.vocab)