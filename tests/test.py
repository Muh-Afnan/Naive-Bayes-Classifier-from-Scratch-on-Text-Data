import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from naive_bayes import NaiveBayes


def test_predict_spam():
    """Test that spam documents are correctly predicted as spam"""
    nb = NaiveBayes()
    X = ["win free money", "free prize click", "meeting tomorrow", "lunch today"]
    y = ["spam", "spam", "not_spam", "not_spam"]
    nb.fit(X, y)
    
    predictions = nb.predict(["win free money"])
    assert predictions[0] == "spam", f"Expected 'spam', got '{predictions[0]}'"


def test_predict_not_spam():
    """Test that ham documents are correctly predicted as not_spam"""
    nb = NaiveBayes()
    X = ["win free money", "free prize click", "meeting tomorrow", "lunch today"]
    y = ["spam", "spam", "not_spam", "not_spam"]
    nb.fit(X, y)
    
    predictions = nb.predict(["meeting tomorrow"])
    assert predictions[0] == "not_spam", f"Expected 'not_spam', got '{predictions[0]}'"


def test_predict_proba_sums_to_one():
    """Test that predicted probabilities for all classes sum to 1"""
    nb = NaiveBayes()
    X = ["win free money", "free prize click", "meeting tomorrow", "lunch today"]
    y = ["spam", "spam", "not_spam", "not_spam"]
    nb.fit(X, y)
    
    proba = nb.predict_proba(["win free money"])
    total_prob = sum(proba[0].values())
    assert abs(total_prob - 1.0) < 1e-6, f"Probabilities sum to {total_prob}, not 1.0"


def test_unseen_word_no_crash():
    """Test that Laplace smoothing handles unseen words without crashing"""
    nb = NaiveBayes(smoothing=1.0)
    X = ["win free money", "free prize click", "meeting tomorrow", "lunch today"]
    y = ["spam", "spam", "not_spam", "not_spam"]
    nb.fit(X, y)
    
    # "xyzabc" is not in training data
    predictions = nb.predict(["xyzabc urgent"])
    assert len(predictions) == 1, "Should return one prediction"
    assert predictions[0] in nb.classes, f"Prediction should be one of the classes"


def test_tokenize_lowercase():
    """Test that tokenization converts text to lowercase"""
    nb = NaiveBayes()
    X = ["HELLO WORLD", "hello world"]
    y = ["class1", "class1"]
    nb.fit(X, y)
    
    # Both should produce the same tokens
    assert "hello" in nb.vocab, "Should have 'hello' in lowercase in vocab"
    assert "world" in nb.vocab, "Should have 'world' in lowercase in vocab"


def test_tokenize_removes_punctuation():
    """Test that tokenization removes punctuation and special characters"""
    nb = NaiveBayes()
    X = ["hello, world!", "hello world"]
    y = ["class1", "class1"]
    nb.fit(X, y)
    
    # Punctuation should be removed
    assert "," not in nb.vocab, "Comma should not be in vocab"
    assert "!" not in nb.vocab, "Exclamation mark should not be in vocab"
    assert "hello" in nb.vocab, "Should have 'hello' in vocab"
    assert "world" in nb.vocab, "Should have 'world' in vocab"


def test_fit_class_counts():
    """Test that class counts are correctly computed during fitting"""
    nb = NaiveBayes()
    X = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    y = ["spam", "spam", "spam", "not_spam", "not_spam"]
    nb.fit(X, y)
    
    assert nb.class_counts["spam"] == 3, f"Expected 3 spam docs, got {nb.class_counts['spam']}"
    assert nb.class_counts["not_spam"] == 2, f"Expected 2 not_spam docs, got {nb.class_counts['not_spam']}"


def test_fit_vocab():
    """Test that vocabulary is correctly built during fitting"""
    nb = NaiveBayes()
    X = ["hello world", "goodbye world"]
    y = ["class1", "class2"]
    nb.fit(X, y)
    
    assert "hello" in nb.vocab, "Should have 'hello' in vocab"
    assert "world" in nb.vocab, "Should have 'world' in vocab"
    assert "goodbye" in nb.vocab, "Should have 'goodbye' in vocab"
    assert len(nb.vocab) == 3, f"Expected vocab size 3, got {len(nb.vocab)}"