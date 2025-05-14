import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.metrics import classification_report
from typing import List, Tuple, Dict, Any


VOCAB_SIZE: int = 10000  
MAX_SEQUENCE_LENGTH: int = 250
EMBEDDING_DIM: int = 128  
LSTM_UNITS: int = 64     
BATCH_SIZE: int = 64
EPOCHS: int = 3 
DATASET_SUBSET_SIZE: int | None = 5000 
RANDOM_STATE: int = 42



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
    text = re.sub(r'<.*?>', '', text)

    return text


def main() -> None:
    """
    Main function to load data, build, train, and evaluate the LSTM sentiment model.
    """
    print("Loading IMDb dataset...")
    # Load the dataset
    (x_train_indices, y_train), (x_test_indices, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    print(f"Original dataset size: Train={len(x_train_indices)}, Test={len(x_test_indices)}")

  
    if DATASET_SUBSET_SIZE is not None:
        print(f"Selecting subset: {DATASET_SUBSET_SIZE} train and {int(DATASET_SUBSET_SIZE * 0.25)} test samples.")
        train_subset_size = min(DATASET_SUBSET_SIZE, len(x_train_indices))
        test_subset_size = min(int(DATASET_SUBSET_SIZE * 0.25), len(x_test_indices))

        x_train_indices = x_train_indices[:train_subset_size]
        y_train = y_train[:train_subset_size]
        x_test_indices = x_test_indices[:test_subset_size]
        y_test = y_test[:test_subset_size]
        print(f"Using subset: Train={len(x_train_indices)}, Test={len(x_test_indices)}")
    else:
        print("Using the full dataset.")

   
    print(f"Padding sequences to max length {MAX_SEQUENCE_LENGTH}...")
    x_train_padded: np.ndarray = pad_sequences(x_train_indices, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    x_test_padded: np.ndarray = pad_sequences(x_test_indices, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    print(f"Padded sequence shape: Train={x_train_padded.shape}, Test={x_test_padded.shape}")

 
    print("Building LSTM model...")
    model: Sequential = Sequential([

        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(units=LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),

        # Output Layer
        Dense(units=1, activation='sigmoid')
    ])

    model.summary()  

    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #  Train the Model
    print("Training model...")
    history: tf.keras.callbacks.History = model.fit(
        x_train_padded,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test_padded, y_test),
        verbose=1 
    )

    # Evaluate the Model 
    print("Evaluating model on test data...")
    loss: float
    accuracy: float
    loss, accuracy = model.evaluate(x_test_padded, y_test, verbose=0)
    print(f"\\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


    print("\\nClassification Report (Neural Network):")
    y_pred_proba: np.ndarray = model.predict(x_test_padded, verbose=0)
    y_pred_classes: np.ndarray = (y_pred_proba > 0.5).astype("int32")
    print(classification_report(y_test, y_pred_classes, target_names=['Negative', 'Positive']))

    print("\\nExample Prediction (Requires word_index mapping):")
    word_index: Dict[str, int] = imdb.get_word_index()

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
        indices: List[int] = [1] 
        indices.extend([word_index.get(word, 2) for word in tokens if word_index.get(word, 2) < vocab_size])

        padded_sequence: np.ndarray = pad_sequences([indices], maxlen=max_len, padding='post', truncating='post')
        return padded_sequence


    processed_pos: np.ndarray = preprocess_text_for_prediction(example_review_positive, word_index, MAX_SEQUENCE_LENGTH, VOCAB_SIZE)
    processed_neg: np.ndarray = preprocess_text_for_prediction(example_review_negative, word_index, MAX_SEQUENCE_LENGTH, VOCAB_SIZE)

    pred_pos_proba: float = model.predict(processed_pos)[0][0]
    pred_neg_proba: float = model.predict(processed_neg)[0][0]

    pred_pos_label: str = "Positive" if pred_pos_proba > 0.5 else "Negative"
    pred_neg_label: str = "Positive" if pred_neg_proba > 0.5 else "Negative"

    print(f"Review: '{example_review_positive}' -> Prediction: {pred_pos_label} (Probability: {pred_pos_proba:.3f})")
    print(f"Review: '{example_review_negative}' -> Prediction: {pred_neg_label} (Probability: {pred_neg_proba:.3f})")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    main() 