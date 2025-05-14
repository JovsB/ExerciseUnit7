import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from typing import List, Tuple, Any

DATASET_SUBSET_SIZE: int | None = 5000
RANDOM_STATE: int = 42

def clean_text(text: str) -> str:
    """
    Cleans the input text by converting to lowercase and removing HTML tags.

    Args:
        text: The input string containing the text review.

    Returns:
        The cleaned text string.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    return text

def main() -> None:
    """
    Main function to load data, train, and evaluate the traditional sentiment model.
    """
    print("Loading IMDb dataset...")
    # Load dataset
    (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.imdb.load_data(
        num_words=None 
    )

    word_index: dict[str, int] = tf.keras.datasets.imdb.get_word_index()
    index_word: dict[int, str] = {v: k for k, v in word_index.items()}

    def decode_review(indices: List[int]) -> str:
        """Decodes a review from word indices back to text."""
        return ' '.join([index_word.get(i - 3, '?') for i in indices if i >= 3])

    print("Decoding reviews...")
    x_train_text_all: List[str] = [decode_review(seq) for seq in x_train_all]
    x_test_text_all: List[str] = [decode_review(seq) for seq in x_test_all] 
 
    y_train_all_labels = np.array(y_train_all)
    y_test_all_labels = np.array(y_test_all)

    print(f"Original dataset size: Train={len(x_train_text_all)}, Test={len(x_test_text_all)}")

    all_texts: List[str] = x_train_text_all + x_test_text_all
    all_labels: np.ndarray = np.concatenate((y_train_all_labels, y_test_all_labels))

    if DATASET_SUBSET_SIZE is not None and DATASET_SUBSET_SIZE < len(all_texts):
        print(f"Using a subset of {DATASET_SUBSET_SIZE} samples.")
      
        indices = np.random.choice(len(all_texts), DATASET_SUBSET_SIZE, replace=False)
        texts_subset: List[str] = [all_texts[i] for i in indices]
        labels_subset: np.ndarray = all_labels[indices]
    else:
        print("Using the full dataset.")
        texts_subset = all_texts
        labels_subset = all_labels

    print("Cleaning text data...")
    texts_cleaned: List[str] = [clean_text(text) for text in texts_subset]


    print("Splitting data into train/test sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        texts_cleaned, labels_subset, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_subset
    )
    print(f"Subset split: Train={len(x_train)}, Test={len(x_test)}")


    print("Applying TF-IDF Vectorizer...")

    vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=10000, stop_words='english')


    x_train_tfidf: Any = vectorizer.fit_transform(x_train)
    x_test_tfidf: Any = vectorizer.transform(x_test) 

    print(f"TF-IDF feature shape: Train={x_train_tfidf.shape}, Test={x_test_tfidf.shape}")
    print("Training Logistic Regression model...")
    
    model: LogisticRegression = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver='liblinear')
    model.fit(x_train_tfidf, y_train)

    print("Evaluating the model...")
    y_pred: np.ndarray = model.predict(x_test_tfidf)


    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    print("\\nExample Prediction:")
    example_review_positive: str = "This movie was fantastic! Great acting and a wonderful plot."
    example_review_negative: str = "Absolutely terrible film. Boring and poorly directed."

    cleaned_pos: str = clean_text(example_review_positive)
    cleaned_neg: str = clean_text(example_review_negative)

    vectorized_pos: Any = vectorizer.transform([cleaned_pos])
    vectorized_neg: Any = vectorizer.transform([cleaned_neg])

    pred_pos: int = model.predict(vectorized_pos)[0]
    pred_neg: int = model.predict(vectorized_neg)[0]

    print(f"Review: '{example_review_positive}' -> Prediction: {'Positive' if pred_pos == 1 else 'Negative'}")
    print(f"Review: '{example_review_negative}' -> Prediction: {'Positive' if pred_neg == 1 else 'Negative'}")


if __name__ == "__main__":
    main() 