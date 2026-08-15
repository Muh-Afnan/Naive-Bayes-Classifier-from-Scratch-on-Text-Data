import matplotlib.pyplot as plt

def plot_word_probabilities(model, class_name, top_n=10):
    """
    Bar chart of top N words by probability for a given class.
    
    Args:
        model: Trained NaiveBayes model
        class_name: The class label to visualize
        top_n: Number of top words to display
    """
    if class_name not in model.classes:
        raise ValueError(f"Class '{class_name}' not found in model classes")
    
    # Get word counts for the class
    word_counts = model.word_counts[class_name]
    
    # Calculate probabilities for each word
    total_words = model.class_word_totals[class_name]
    word_probs = {word: count / total_words for word, count in word_counts.items()}
    
    # Sort by probability and get top N
    sorted_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, probs = zip(*sorted_words)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(words, probs, color='steelblue')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Top {top_n} Words by Probability for Class: {class_name}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y):
    """
    Bar chart of class counts in training data.
    
    Args:
        y: List of class labels
    """
    # Count occurrences of each class
    class_counts = {}
    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    # Create bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(classes, counts, color='coral')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution in Training Data', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    2x2 grid showing TP, FP, TN, FN for binary classification.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
    """
    # Get unique classes
    classes = sorted(set(y_true) | set(y_pred))
    
    if len(classes) != 2:
        raise ValueError("Confusion matrix visualization only supports binary classification")
    
    pos_class, neg_class = classes[1], classes[0]
    
    # Calculate confusion matrix values
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pos_class and pred == pos_class)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == neg_class and pred == pos_class)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == neg_class and pred == neg_class)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == pos_class and pred == neg_class)
    
    # Create confusion matrix grid
    cm = [[tn, fp], [fn, tp]]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add text annotations
    plt.text(0, 0, f'TN\n{tn}', ha='center', va='center', fontsize=14, color='white', weight='bold')
    plt.text(1, 0, f'FP\n{fp}', ha='center', va='center', fontsize=14, color='white', weight='bold')
    plt.text(0, 1, f'FN\n{fn}', ha='center', va='center', fontsize=14, color='white', weight='bold')
    plt.text(1, 1, f'TP\n{tp}', ha='center', va='center', fontsize=14, color='white', weight='bold')
    
    # Set labels and ticks
    plt.xticks([0, 1], [f'Predicted\n{neg_class}', f'Predicted\n{pos_class}'])
    plt.yticks([0, 1], [f'Actual {neg_class}', f'Actual {pos_class}'])
    plt.title('Confusion Matrix', fontsize=14, weight='bold')
    plt.colorbar()
    plt.tight_layout()
    plt.show()