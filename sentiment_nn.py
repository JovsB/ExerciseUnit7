import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.metrics import classification_report

# --- Type Definitions ---
from typing import List, Tuple, Dict, Any

# --- Constants ---
VOCAB_SIZE: int = 10000  # Max number of words to keep in the vocabulary
MAX_SEQUENCE_LENGTH: int = 250  # Max length of review sequences (pad/truncate)
EMBEDDING_DIM: int = 128  # Dimension of word embeddings
LSTM_UNITS: int = 64     # Number of units in the LSTM layer
BATCH_SIZE: int = 64
EPOCHS: int = 3 # Keep epochs low for faster demonstration
DATASET_SUBSET_SIZE: int | None = 5000 # Use subset for speed, None for full dataset
RANDOM_STATE: int = 42

# --- Helper Functions ---

def clean_text(text: str) -> str:
    """
    Cleans the input text by converting to lowercase and removing HTML tags.
    Handles potential decoding issues if text is bytes.

    Args:
        text: The input string or bytes containing the text review.

    Returns:
        The cleaned text string.
    """
    # Ensure text is string
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')

    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters (optional)
    # text = re.sub(r'[^a-z0-9\\s]', '', text)
    return text

# --- Main Execution ---
def main() -> None:
    """
    Main function to load data, build, train, and evaluate the LSTM sentiment model.
    """
    print("Loading IMDb dataset...")
    # Load the dataset, keeping only the top VOCAB_SIZE words
    (x_train_indices, y_train), (x_test_indices, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    print(f"Original dataset size: Train={len(x_train_indices)}, Test={len(x_test_indices)}")

    # --- Data Selection (Optional Subset) ---
    if DATASET_SUBSET_SIZE is not None:
        print(f"Selecting subset: {DATASET_SUBSET_SIZE} train and {int(DATASET_SUBSET_SIZE * 0.25)} test samples.")
        # Simple slicing for subset selection
        train_subset_size = min(DATASET_SUBSET_SIZE, len(x_train_indices))
        test_subset_size = min(int(DATASET_SUBSET_SIZE * 0.25), len(x_test_indices))

        x_train_indices = x_train_indices[:train_subset_size]
        y_train = y_train[:train_subset_size]
        x_test_indices = x_test_indices[:test_subset_size]
        y_test = y_test[:test_subset_size]
        print(f"Using subset: Train={len(x_train_indices)}, Test={len(x_test_indices)}")
    else:
        print("Using the full dataset.")

    # --- Padding Sequences ---
    print(f"Padding sequences to max length {MAX_SEQUENCE_LENGTH}...")
    x_train_padded: np.ndarray = pad_sequences(x_train_indices, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    x_test_padded: np.ndarray = pad_sequences(x_test_indices, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    print(f"Padded sequence shape: Train={x_train_padded.shape}, Test={x_test_padded.shape}")

    # --- Build the LSTM Model ---
    print("Building LSTM model...")
    model: Sequential = Sequential([
        # 1. Embedding Layer: Turns positive integers (indexes) into dense vectors of fixed size.
        # input_dim is the size of the vocabulary plus one for potential padding/OOV if not handled by TextVectorization.
        # Since load_data uses indices starting from 1 (0 reserved), vocab size is appropriate.
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),

        # 2. LSTM Layer: Processes sequences, capturing temporal dependencies.
        # dropout and recurrent_dropout help prevent overfitting.
        LSTM(units=LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),

        # 3. Output Layer: A single neuron with sigmoid activation for binary classification.
        Dense(units=1, activation='sigmoid')
    ])

    model.summary() # Print model architecture

    # --- Compile the Model ---
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # --- Train the Model ---
    print("Training model...")
    history: tf.keras.callbacks.History = model.fit(
        x_train_padded,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test_padded, y_test),
        verbose=1 # Show progress
    )

    # --- Evaluate the Model ---
    print("Evaluating model on test data...")
    loss: float
    accuracy: float
    loss, accuracy = model.evaluate(x_test_padded, y_test, verbose=0)
    print(f"\\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Add Classification Report
    print("\\nClassification Report (Neural Network):")
    y_pred_proba: np.ndarray = model.predict(x_test_padded, verbose=0)
    # Convert probabilities to class labels (0 or 1) based on a 0.5 threshold
    y_pred_classes: np.ndarray = (y_pred_proba > 0.5).astype("int32")
    # Ensure y_test is in the correct format if it's not already (e.g., 1D array)
    # imdb.load_data already provides y_test as a 1D array of labels.
    print(classification_report(y_test, y_pred_classes, target_names=['Negative', 'Positive']))

    # --- Example Prediction ---
    # Note: For real examples, we need the same preprocessing (tokenization, padding)
    # This requires the word_index used by imdb.load_data
    print("\\nExample Prediction (Requires word_index mapping):")
    word_index: Dict[str, int] = imdb.get_word_index()
    # Add special tokens (padding, start, unknown) which imdb.load_data uses internally
    # Indices are offset by 3 in the loaded data
    index_word: Dict[int, str] = {i + 3: word for word, i in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"

    example_review_positive: str = "This movie was fantastic! Great acting and a wonderful plot."
    example_review_negative: str = "Absolutely terrible film. Boring and poorly directed."

    def preprocess_text_for_prediction(text: str, word_index: Dict[str, int], max_len: int, vocab_size: int) -> np.ndarray:
        """Preprocesses a single text review for prediction."""
        cleaned: str = clean_text(text)
        tokens: List[str] = cleaned.split()
        # Convert words to indices, using <UNK> index (2) for out-of-vocabulary words
        # Also respect the vocab_size limit used during loading
        indices: List[int] = [1] # Start with <START> token index
        indices.extend([word_index.get(word, 2) for word in tokens if word_index.get(word, 2) < vocab_size])

        # Pad the sequence
        padded_sequence: np.ndarray = pad_sequences([indices], maxlen=max_len, padding='post', truncating='post')
        return padded_sequence

    # Preprocess example reviews
    processed_pos: np.ndarray = preprocess_text_for_prediction(example_review_positive, word_index, MAX_SEQUENCE_LENGTH, VOCAB_SIZE)
    processed_neg: np.ndarray = preprocess_text_for_prediction(example_review_negative, word_index, MAX_SEQUENCE_LENGTH, VOCAB_SIZE)

    # Make predictions
    pred_pos_proba: float = model.predict(processed_pos)[0][0]
    pred_neg_proba: float = model.predict(processed_neg)[0][0]

    pred_pos_label: str = "Positive" if pred_pos_proba > 0.5 else "Negative"
    pred_neg_label: str = "Positive" if pred_neg_proba > 0.5 else "Negative"

    print(f"Review: '{example_review_positive}' -> Prediction: {pred_pos_label} (Probability: {pred_pos_proba:.3f})")
    print(f"Review: '{example_review_negative}' -> Prediction: {pred_neg_label} (Probability: {pred_neg_proba:.3f})")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    main() 