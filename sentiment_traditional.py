import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

# --- Type Definitions ---
from typing import List, Tuple, Any

# --- Constants ---
# Using a smaller subset for faster demonstration
# Set to None to use the full dataset (50,000 reviews)
DATASET_SUBSET_SIZE: int | None = 5000
RANDOM_STATE: int = 42

# --- Helper Functions ---

def clean_text(text: str) -> str:
    """
    Cleans the input text by converting to lowercase and removing HTML tags.

    Args:
        text: The input string containing the text review.

    Returns:
        The cleaned text string.
    """
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters (optional, could keep punctuation)
    # text = re.sub(r'[^a-z0-9\\s]', '', text)
    return text

# --- Main Execution ---

def main() -> None:
    """
    Main function to load data, train, and evaluate the traditional sentiment model.
    """
    print("Loading IMDb dataset...")
    # Load dataset (already split into train/test)
    (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.imdb.load_data(
        num_words=None # Keep all words initially for TF-IDF
    )

    # The loaded data is sequences of word indices. We need the original text.
    # Get the word index mapping
    word_index: dict[str, int] = tf.keras.datasets.imdb.get_word_index()
    # Reverse the word index to map indices back to words
    index_word: dict[int, str] = {v: k for k, v in word_index.items()}

    # Function to decode reviews
    def decode_review(indices: List[int]) -> str:
        """Decodes a review from word indices back to text."""
        # Index 0 is padding, 1 is start of sequence, 2 is unknown
        # We shift indices by 3 and handle unknowns
        return ' '.join([index_word.get(i - 3, '?') for i in indices if i >= 3])

    print("Decoding reviews...")
    x_train_text_all: List[str] = [decode_review(seq) for seq in x_train_all]
    x_test_text_all: List[str] = [decode_review(seq) for seq in x_test_all] # Corrected: Use x_test_all for indices
    # Note: The original imdb.load_data provides labels directly, no need to decode y
    y_train_all_labels = np.array(y_train_all)
    y_test_all_labels = np.array(y_test_all) # Use original y_test labels

    print(f"Original dataset size: Train={len(x_train_text_all)}, Test={len(x_test_text_all)}")

    # Combine and select subset if specified
    all_texts: List[str] = x_train_text_all + x_test_text_all
    all_labels: np.ndarray = np.concatenate((y_train_all_labels, y_test_all_labels))

    if DATASET_SUBSET_SIZE is not None and DATASET_SUBSET_SIZE < len(all_texts):
        print(f"Using a subset of {DATASET_SUBSET_SIZE} samples.")
        # Ensure stratification if possible, though simple slicing is used here for speed
        indices = np.random.choice(len(all_texts), DATASET_SUBSET_SIZE, replace=False)
        texts_subset: List[str] = [all_texts[i] for i in indices]
        labels_subset: np.ndarray = all_labels[indices]
    else:
        print("Using the full dataset.")
        texts_subset = all_texts
        labels_subset = all_labels

    print("Cleaning text data...")
    texts_cleaned: List[str] = [clean_text(text) for text in texts_subset]

    # Split data into training and testing sets
    print("Splitting data into train/test sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        texts_cleaned, labels_subset, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_subset
    )
    print(f"Subset split: Train={len(x_train)}, Test={len(x_test)}")


    print("Applying TF-IDF Vectorizer...")
    # Create TF-IDF vectorizer
    # max_features can be tuned; limits the vocabulary size
    vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

    # Fit on training data and transform both train and test data
    x_train_tfidf: Any = vectorizer.fit_transform(x_train)
    x_test_tfidf: Any = vectorizer.transform(x_test) # Use transform, not fit_transform, on test data

    print(f"TF-IDF feature shape: Train={x_train_tfidf.shape}, Test={x_test_tfidf.shape}")

    print("Training Logistic Regression model...")
    # Initialize and train the Logistic Regression model
    # C is the inverse of regularization strength; smaller values specify stronger regularization.
    # max_iter might need adjustment depending on the dataset size and convergence.
    model: LogisticRegression = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver='liblinear')
    model.fit(x_train_tfidf, y_train)

    print("Evaluating the model...")
    # Make predictions on the test set
    y_pred: np.ndarray = model.predict(x_test_tfidf)

    # Print the classification report
    print("\\nClassification Report:")
    # target_names map label indices (0, 1) to human-readable names
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # Example prediction
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
    # Download necessary NLTK data if using stop_words='english' or other nltk features
    # try:
    #     import nltk
    #     nltk.data.find('corpora/stopwords')
    # except nltk.downloader.DownloadError:
    #     print("Downloading 'stopwords' corpus from NLTK...")
    #     nltk.download('stopwords')
    # except ImportError:
    #      print("NLTK not installed, skipping stopword download check.")

    # Ensure TensorFlow dataset caching directory exists or handle potential errors
    # (TensorFlow usually handles this automatically)

    main() 